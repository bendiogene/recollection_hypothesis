import numpy as np
from math import floor, ceil
from os import listdir
from sys import exit
from itertools import chain, tee
from numba import cuda
from cuda_utils import *
from DoG_filt_cuda import *
from cpu_utils import *
import numba as nb
import pickle
import os
from timeit import default_timer as timer
import random
np.set_printoptions(threshold = np.inf)

class SDNN:
    """ 
        __author__ = Nicolas Perez-Nieves
        __email__ = nicolas.perez14@imperial.ac.uk
        
        This class implements a STDP-based Spiking Convolutional Deep Neural Network 
        for image or video recognition. This implementation is based on the implementation on [1]

        The input consists of up to M_in channels where the information on each channel 
        is coded in the spike times following a rank-order coding. 
        The input, of size H_in x W_in x M_in, passes through a set of convolutional-pooling
        layers which extract the features of the image.

        The training is done on each convolutional layer in an unsupervised manner following 
        an STDP rule. Only the convolution layer weights are updated following this rule.
        There is no training in the pooling layers.

        The neurons used are non-leaky integrate-and-fire (NL-IAF). The voltage V of the 
        neurons follows:

        V_i(t) = V_i(t-1)+ sum_j( W_i,j * S_j(t-1))         [1]

        Where i and j correspond to post-synaptic and pre-synaptic neurons respectively.
        S are the spikes times from the previous layer.

        After every voltage update (each time step) the weights are updated following:

        dw(i, j) = a_plus * w(i, j) * (1-w(i, j)) if t_j-t_i <= 0
        dw(i, j) = a_minus * w(i, j) * (1-w(i, j)) if t_j-t_i > 0        [1]

       where i and j correspond to post and pre synaptic neurons respectively and a is 
       the learning rate.
       Note that weights will always be in the interval [0, 1].

       The learning is done layer by layer. No layer will learn until the previous has 
       finished learning.


       References:
       [1] Kheradpisheh, S.R., et al. STDP-based spiking deep neural networks for object recognition.
            arXiv:1611.01421v1 (Nov, 2016)
            
       -------------------------
       _Modified by Andrea Tomassilli and Zied Ben Houidi
        _ Last layer first modified by Andrea Tomassilli, later by Zied Ben Houidi to account for backpropagation based recollection
    """

    def __init__(self, network_params, weight_params, stdp_params, total_time, DoG_params=None,
                 spike_times_learn=None, spike_times_train=None, spike_times_test=None,
                 y_train=None, y_test=None, device='GPU'):
        """
            Initialisaition of SDNN

            Input:            
            - network_params: A list of dictionaries with the following keys:                
                -'Type': A string specifying which kind of layer this is (either 'input', 'conv' and 'pool')
                -'num_filters': an int specifying the depth (number of filters) of this layer
                -'filter_size': an int specifying the height and width of the filter window for 
                                the previous layer to this layer (only on 'conv' and  'pool')
                -'th': an np.float32 specifying the threshold of this layer (only on 'conv' and  'pool')
                -'stride': an int specifying the stride for this layer (only on 'pool')
                -'pad': an int specifying the pad for this layer (only on 'input')
                -'H_layer': an int specifying the height of this layer (only on 'input')
                -'W_layer': an int specifying the width of this layer (only on 'input') 
            - weight_params: A dictionary with the following keys:                
                - 'mean': the mean for initialising the weights
                - 'std': the std for initialising the weights
            - stdp_params: A dictionary with the following keys:                                
                - 'max_iter': an int specifyng the maximum number of iterations allowed on learning
                - 'max_learn_iter': a list of ints specifying the maximum number of iterations allowed for training each layer (len = number of layers)
                - 'stdp_per_layer': a list of ints specifying the maximum number of STDP updates per layer (len = number of layers)
                - 'offset_STDP': a list of ints specifying the STDP ofset per leayer updates per layer (len = number of layers)
                - 'a_minus': an np.float32 numpy array specifying the learning rate when no causality 
                - 'a_plus': an np.float32 numpy array specifying the learning rate when there is causality 
            - total_time: An int specifying the number of time steps per image
            - spike_times_learn: A list of strings with a valid absolute or relative path to the folders with 
                                 the learning .jpg images OR 
                                 An uint8 array with the learning spike times of shape (N_lr, H_in, W_in, M_in). 
                                 Axis 0 is each of the images
            - spike_times_train: A list of strings with a valid absolute or relative path to the folders with 
                                 the training .jpg images OR 
                                 An uint8 array with the training spike times of shape (N_tr, H_in, W_in, M_in). 
                                 Axis 0 is each of the images
            - spike_times_test: A list of strings with a valid absolute or relative path to the folders with 
                                 the testing .jpg images OR 
                                 An uint8 array with the testing spike times of shape (N_ts, H_in, W_in, M_in). 
                                 Axis 0 is each of the images   
            - DoG_params: None OR A dictionary with the following keys:
                -'img_size': A tuple of integers with the dimensions to which the images are to be resized 
                -'DoG_size': An int with the size of the DoG filter window size
                -'std1': A float with the standard deviation 1 for the DoG filter
                -'std2': A float with the standard deviation 2 for the DoG filter                  
                 
        """

        # --------------------------- DoG Filter Parameters -------------------#
        if DoG_params is not None:
            self.DoG = True
            self.img_size = DoG_params['img_size']
            self.filt = DoG(DoG_params['DoG_size'], DoG_params['std1'], DoG_params['std2'])
        else:
            self.DoG = False

        # --------------------------- Network Initialisation -------------------#
        # Total time and number of layers
        self.num_layers = len(network_params)
        self.learnable_layers = []
        self.total_time = total_time

        # Layers Initialisation
        self.network_struc = []
        self.init_net_struc(network_params)
        self.layers = []
        self.init_layers()

        # Weights Initialisation
        self.weight_params = weight_params
        self.weights = []
        self.init_weights()

        # Dimension Check
        self.check_dimensions()

        # ---------------------------Learning Paramters -------------------#
        # Learning layer parameters
        self.max_iter = stdp_params['max_iter']
        self.learning_layer = self.learnable_layers[0]
        self.max_learn_iter = stdp_params['max_learn_iter']
        self.curr_lay_idx = 0
        self.counter = 0
        self.curr_img = 0

        #STDP params
        self.stdp_per_layer = stdp_params['stdp_per_layer']
        self.stdp_a_minus = stdp_params['a_minus']
        self.stdp_a_plus = stdp_params['a_plus']
        self.offsetSTDP = stdp_params['offset_STDP']

        # --------------------------- CUDA Parameters -------------------#
        self.device = device
        if self.device == 'GPU':
            self.thds_per_dim = 10  # (Use 8 if doesn't work)

        # --------------------------- Input spike times -------------------#
        # Generate Iterators with the full path to the images in each set OR reference the spike times
        if self.DoG:
            self.spike_times_learn, self.y_learn = self.gen_iter_paths(spike_times_learn)
            self.spike_times_train, self.y_train = self.gen_iter_paths(spike_times_train)
            self.spike_times_test, self.y_test = self.gen_iter_paths(spike_times_test)
            self.num_img_learn = self.y_learn.size
            self.num_img_train = self.y_train.size
            self.num_img_test = self.y_test.size

            self.spike_times_train, self.learn_buffer, self.train_score_img = tee(self.spike_times_train,3)

        else:
            self.spike_times_learn = spike_times_learn
            self.num_img_learn = spike_times_learn.shape[0]
            self.spike_times_train = spike_times_train
            self.num_img_train = spike_times_train.shape[0]
            self.spike_times_test = spike_times_test
            self.num_img_test = spike_times_test.shape[0]
            self.y_train = y_train
            self.y_test = y_test

        # --------------------------- Output features -------------------#
        self.features_train = []
        self.features_test = []


# --------------------------- Initialisation functions ------------------------#
    # Network Structure Initialization
    def init_net_struc(self, network_params):
        """
            Network structure initialisation 
        """

        for i in range(self.num_layers):
            d_tmp = {}
            if network_params[i]['Type'] == 'input':
                d_tmp['Type'] = network_params[i]['Type']
                d_tmp['H_layer'] = network_params[i]['H_layer']
                d_tmp['W_layer'] = network_params[i]['W_layer']
                d_tmp['num_filters'] = network_params[i]['num_filters']
                d_tmp['pad'] = network_params[i]['pad']
                d_tmp['shape'] = (d_tmp['H_layer'], d_tmp['W_layer'], d_tmp['num_filters'])
            elif network_params[i]['Type'] == 'conv':
                d_tmp['Type'] = network_params[i]['Type']
                d_tmp['th'] = network_params[i]['th']
                d_tmp['filter_size'] = network_params[i]['filter_size']
                d_tmp['num_filters'] = network_params[i]['num_filters']
                d_tmp['pad'] = np.array([int(floor(d_tmp['filter_size']/2)), int(floor(d_tmp['filter_size']/2))])
                d_tmp['stride'] = 1
                d_tmp['offset'] = floor(d_tmp['filter_size']/2)
                d_tmp['H_layer'] = int(1 + floor((self.network_struc[i-1]['H_layer']+2*d_tmp['pad'][0]-d_tmp['filter_size'])/d_tmp['stride']))
                d_tmp['W_layer'] = int(1 + floor((self.network_struc[i-1]['W_layer']+2*d_tmp['pad'][1]-d_tmp['filter_size'])/d_tmp['stride']))
                d_tmp['shape'] = (d_tmp['H_layer'], d_tmp['W_layer'], d_tmp['num_filters'])
                self.learnable_layers.append(i)
            elif network_params[i]['Type'] == 'pool':
                d_tmp['Type'] = network_params[i]['Type']
                d_tmp['th'] = network_params[i]['th']
                d_tmp['filter_size'] = network_params[i]['filter_size']
                d_tmp['num_filters'] = network_params[i]['num_filters']
                d_tmp['pad'] = [int(floor(d_tmp['filter_size']/2)), int(floor(d_tmp['filter_size']/2))]
                d_tmp['stride'] = network_params[i]['stride']
                d_tmp['offset'] = floor(d_tmp['filter_size']/2)
                d_tmp['H_layer'] = int(1 + floor((self.network_struc[i-1]['H_layer']+2*d_tmp['pad'][0]-d_tmp['filter_size'])/d_tmp['stride']))
                d_tmp['W_layer'] = int(1 + floor((self.network_struc[i-1]['W_layer']+2*d_tmp['pad'][1]-d_tmp['filter_size'])/d_tmp['stride']))
                d_tmp['shape'] = (d_tmp['H_layer'], d_tmp['W_layer'], d_tmp['num_filters'])
            else:
                exit("unknown layer specified: use 'input', 'conv' or 'pool' ")
            self.network_struc.append(d_tmp)

    # Weights Initialization
    def init_weights(self):
        """
            Weight Initialization
        """
        mean = self.weight_params['mean']
        std = self.weight_params['std']
        for i in range(1, self.num_layers):
            HH = self.network_struc[i]['filter_size']
            WW = self.network_struc[i]['filter_size']
            MM = self.network_struc[i - 1]['num_filters']
            DD = self.network_struc[i]['num_filters']
            w_shape = (HH, WW, MM, DD)
            if self.network_struc[i]['Type'] == 'conv':
                weights_tmp = (mean + std * np.random.normal(size=w_shape))
                weights_tmp[weights_tmp >= 1.] = 0.99
                weights_tmp[weights_tmp <= 0.] = 0.01
            elif self.network_struc[i]['Type'] == 'pool':
                weights_tmp = np.ones((HH, WW, MM))/(HH*WW)
            else:
                continue
            self.weights.append(weights_tmp.astype(np.float32))

    # Dimension Checker
    def check_dimensions(self):
        """
            Checks the dimensions of the SDNN
        """
        for i in range(1, self.num_layers):
            H_pre, W_pre, M_pre = self.network_struc[i - 1]['shape']
            if self.network_struc[i]['Type'] == 'conv':
                HH, WW, MM, DD = self.weights[i-1].shape
            else:
                HH, WW, MM = self.weights[i-1].shape
            H_post, W_post, D_post = self.network_struc[i]['shape']
            stride = self.network_struc[i]['stride']
            H_pad, W_pad = self.network_struc[i]['pad']

            assert floor((H_pre + 2*H_pad - HH) / stride) + 1 == H_post, 'Error HEIGHT: layer %s to layer %s . ' \
                                                                    'Width does not work' % (i-1,  i)
            assert floor((W_pre + 2*W_pad - WW) / stride) + 1 == W_post, 'Error WIDTH: layer %s to layer %s . ' \
                                                                    'Width does not work' % (i-1,  i)
            assert MM == M_pre, 'Error in DEPTH of PREVIOUS map'
            if self.network_struc[i]['Type'] == 'conv':
                assert DD == D_post, 'Error in DEPTH of CURRENT map'

    # Initialise layers
    def init_layers(self):
        """
            Initialise layers         
        """
        for i in range(self.num_layers):
            d_tmp = {}
            H, W, D = self.network_struc[i]['shape']
            d_tmp['S'] = np.zeros((H, W, D, self.total_time)).astype(np.uint8)
            d_tmp['V'] = np.zeros((H, W, D, self.total_time)).astype(np.float32)
            d_tmp['K_STDP'] = np.ones((H, W, D)).astype(np.uint8)
            d_tmp['K_inh'] = np.ones((H, W)).astype(np.uint8)
            self.layers.append(d_tmp)
        return

    # Layers reset
    def reset_layers(self):
        """
            Reset layers         
        """
        for i in range(self.num_layers):
            H, W, D = self.network_struc[i]['shape']
            self.layers[i]['S'] = np.zeros((H, W, D, self.total_time)).astype(np.uint8)
            self.layers[i]['V'] = np.zeros((H, W, D, self.total_time)).astype(np.float32)
            self.layers[i]['K_STDP'] = np.ones((H, W, D)).astype(np.uint8)
            self.layers[i]['K_inh'] = np.ones((H, W)).astype(np.uint8)
        return

    # Weights getter
    def get_weights(self):
        return self.weights

    # Weights setter
    def set_weights(self, path_list):
        """
            Sets the weights to the values specified in path_list

            Input:
            - path_list: A list of strings specifying the addresses to the weights to be set. These weights must be 
                         stored as *.npy                    
        """
        self.weights = []
        for id in range(self.num_layers-1):
            weight_tmp = np.load(path_list[id])
            self.weights.append(weight_tmp.astype(np.float32))
        return

    def gen_iter_paths(self, path_list):
        labels = np.ones(len(listdir(path_list[0])))
        paths_iter = iter([path_list[0] + listdir(path_list[0])[i] for i in range(labels.size)])
        for idir in range(1, len(path_list)):
            file_names = listdir(path_list[idir])
            labels = np.append(labels, (idir+1)*np.ones(len(file_names)))
            files_tmp = iter([path_list[idir] + file_names[i] for i in range(len(file_names))])
            paths_iter = chain(paths_iter, files_tmp)
        return paths_iter, labels

# --------------------------- STDP Learning functions ------------------------#
    # Propagate and STDP once
    def train_step(self):
        """
            Propagates one image through the SDNN network and carries out the STDP update on the learning layer
        """

        # Propagate
        for t in range(1, self.total_time):
            for i in range(1, self.learning_layer+1):

                H, W, D = self.network_struc[i]['shape']
                H_pad, W_pad = self.network_struc[i]['pad']
                stride = self.network_struc[i]['stride']
                th = self.network_struc[i]['th']

                w = self.weights[i-1]
                s = self.layers[i - 1]['S'][:, :, :, t - 1]  # Input spikes
                s = np.pad(s, ((H_pad, H_pad), (W_pad, W_pad), (0, 0)), mode='constant')  # Pad the input
                S = self.layers[i]['S'][:, :, :, t]  # Output spikes
                V = self.layers[i]['V'][:, :, :, t - 1]  # Output voltage before
                K_inh = self.layers[i]['K_inh']  # Lateral inhibition matrix

                blockdim = (self.thds_per_dim, self.thds_per_dim, self.thds_per_dim)
                griddim = (int(ceil(H / blockdim[0])) if int(ceil(H / blockdim[2])) != 0 else 1,
                           int(ceil(W / blockdim[1])) if int(ceil(W / blockdim[2])) != 0 else 1,
                           int(ceil(D / blockdim[2])) if int(ceil(D / blockdim[2])) != 0 else 1)

                if self.network_struc[i]['Type'] == 'conv':
                    V, S = self.convolution(S, V, s, w, stride, th, blockdim, griddim)
                    self.layers[i]['V'][:, :, :, t] = V

                    S, K_inh = self.lateral_inh(S, V, K_inh, blockdim, griddim)
                    self.layers[i]['S'][:, :, :, t] = S
                    self.layers[i]['K_inh'] = K_inh

                elif self.network_struc[i]['Type'] == 'pool':
                    S = self.pooling(S, s, w, stride, th, blockdim, griddim)
                    self.layers[i]['S'][:, :, :, t] = S

                    if i < 3:
                        S, K_inh = self.lateral_inh(S, V, K_inh, blockdim, griddim)
                        self.layers[i]['S'][:, :, :, t] = S
                        self.layers[i]['K_inh'] = K_inh

            # STDP learning
            lay = self.learning_layer
            if self.network_struc[lay]['Type'] == 'conv':

                # valid are neurons in the learning layer that can do STDP and that have fired in the current t
                S = self.layers[lay]['S'][:, :, :, t]  # Output spikes
                V = self.layers[lay]['V'][:, :, :, t]  # Output voltage
                K_STDP = self.layers[lay]['K_STDP']  # Lateral inhibition matrix
                valid = S*V*K_STDP

                if np.count_nonzero(valid) > 0:

                    H, W, D = self.network_struc[lay]['shape']
                    stride = self.network_struc[lay]['stride']
                    offset = self.offsetSTDP[lay]
                    a_minus = self.stdp_a_minus[lay]
                    a_plus = self.stdp_a_plus[lay]

                    s = self.layers[lay - 1]['S'][:, :, :, :t]  # Input spikes
                    ssum = np.sum(s, axis=3)
                    s = np.pad(ssum, ((H_pad, H_pad), (W_pad, W_pad), (0, 0)), mode='constant')  # Pad the input
                    w = self.weights[lay - 1]

                    maxval, maxind1, maxind2 = self.get_STDP_idxs(valid, H, W, D, lay)

                    blockdim = (self.thds_per_dim, self.thds_per_dim, self.thds_per_dim)
                    griddim = (int(ceil(H / blockdim[0])) if int(ceil(H / blockdim[2])) != 0 else 1,
                               int(ceil(W / blockdim[1])) if int(ceil(W / blockdim[2])) != 0 else 1,
                               int(ceil(D / blockdim[2])) if int(ceil(D / blockdim[2])) != 0 else 1)

                    w, K_STDP = self.STDP(S.shape, s, w, K_STDP,
                                          maxval, maxind1, maxind2,
                                          stride, offset, a_minus, a_plus, blockdim, griddim)
                    self.weights[lay - 1] = w
                    self.layers[lay]['K_STDP'] = K_STDP



    # Train all images in training set
    def train_SDNN(self):
        """
            Trains the SDNN with the learning set of images
            
            We iterate over the set of images a maximum of self.max_iter times
        """

        print("-----------------------------------------------------------")
        print("-------------------- STARTING LEARNING---------------------")
        print("-----------------------------------------------------------")
        for i in range(self.max_iter):
            print("----------------- Learning Progress  {}%----------------------".format(str(i) + '/'
                                                                                          + str(self.max_iter)
                                                                                          + ' ('
                                                                                          + str(100 * i / self.max_iter)
                                                                                          + ')'))
            if self.counter > self.max_learn_iter[self.learning_layer]:
                self.curr_lay_idx += 1
                self.learning_layer = self.learnable_layers[self.curr_lay_idx]
                self.counter = 0
            self.counter += 1

            self.reset_layers()  # Reset all layers for the new image
            if self.DoG:
                try:
                    path_img = next(self.learn_buffer)
                except:
                    self.spike_times_test, self.learn_buffer = tee(self.spike_times_test)
                    path_img = next(self.learn_buffer)
                st = DoG_filter(path_img, self.filt, self.img_size, self.total_time, self.num_layers)
                st = np.expand_dims(st, axis=2)
            else:
                st = self.spike_times_learn[self.curr_img, :, :, :, :]  # (Image_number, H, W, M, time) to (H, W, M, time)
            self.layers[0]['S'] = st  # (H, W, M, time)
            self.train_step()

            if i % 500 == 0:
                self.stdp_a_plus[self.learning_layer] = min(2.*self.stdp_a_plus[self.learning_layer], 0.15)
                self.stdp_a_minus[self.learning_layer] = 0.75*self.stdp_a_plus[self.learning_layer]

            if self.curr_img+1 < self.num_img_learn:
                self.curr_img += 1
            else:
                self.curr_img = 0
        print("----------------- Learning Progress  {}%----------------------".format(str(self.max_iter) + '/'
                                                                                      + str(self.max_iter)
                                                                                      + ' ('
                                                                                      + str(100)
                                                                                      + ')'))
        print("-----------------------------------------------------------")
        print("------------------- LEARNING COMPLETED --------------------")
        print("-----------------------------------------------------------")

    # Find STDP update indices and potentials
    def get_STDP_idxs(self, valid, H, W, D, layer_idx):
        """
            Finds the indices and potentials of the post-synaptic neurons to update. 
            Only one update per map (if allowed) 
        """

        i = layer_idx
        STDP_counter = 1

        mxv = np.amax(valid, axis=2)
        mxi = np.argmax(valid, axis=2)

        maxind1 = np.ones((D, 1)) * -1
        maxind2 = np.ones((D, 1)) * -1
        maxval = np.ones((D, 1)) * -1

        while np.sum(np.sum(mxv)) != 0.:
            # for each layer a certain number of neurons can do the STDP per image
            if STDP_counter > self.stdp_per_layer[i]:
                break
            else:
                STDP_counter += 1

            maximum = np.amax(mxv, axis=1)
            index = np.argmax(mxv, axis=1)

            index1 = np.argmax(maximum)
            index2 = index[index1]

            maxval[mxi[index1, index2]] = mxv[index1, index2]
            maxind1[mxi[index1, index2]] = index1
            maxind2[mxi[index1, index2]] = index2

            mxv[mxi == mxi[index1, index2]] = 0
            mxv[max(index1 - self.offsetSTDP[layer_idx], 0):min(index1 + self.offsetSTDP[layer_idx], H) + 1,
                max(index2 - self.offsetSTDP[layer_idx], 0):min(index2 + self.offsetSTDP[layer_idx], W) + 1] = 0

        maxval = np.squeeze(maxval).astype(np.float32)
        maxind1 = np.squeeze(maxind1).astype(np.int16)
        maxind2 = np.squeeze(maxind2).astype(np.int16)

        return maxval, maxind1, maxind2

# --------------------------- Propagation functions ------------------------#
    # Propagate once
    def prop_step(self):
        """
            Propagates one image through the SDNN network. 
            This function is identical to train_step() but here  no STDP takes place and we always reach the last layer
        """

        # Propagate
        for t in range(1, self.total_time):
            for i in range(1, self.num_layers):

                H, W, D = self.network_struc[i]['shape']
                H_pad, W_pad = self.network_struc[i]['pad']
                stride = self.network_struc[i]['stride']
                th = self.network_struc[i]['th']

                w = self.weights[i-1]
                s = self.layers[i - 1]['S'][:, :, :, t - 1]  # Input spikes
                s = np.pad(s, ((H_pad, H_pad), (W_pad, W_pad), (0, 0)), mode='constant')  # Pad the input
                S = self.layers[i]['S'][:, :, :, t]  # Output spikes
                V = self.layers[i]['V'][:, :, :, t - 1]  # Output voltage before
                K_inh = self.layers[i]['K_inh']  # Lateral inhibition matrix

                blockdim = (self.thds_per_dim, self.thds_per_dim, self.thds_per_dim)
                griddim = (int(ceil(H / blockdim[0])) if int(ceil(H / blockdim[2])) != 0 else 1,
                           int(ceil(W / blockdim[1])) if int(ceil(W / blockdim[2])) != 0 else 1,
                           int(ceil(D / blockdim[2])) if int(ceil(D / blockdim[2])) != 0 else 1)

                if self.network_struc[i]['Type'] == 'conv':
                    if self.device == 'GPU':
                        V, S = self.convolution(S, V, s, w, stride, th, blockdim, griddim)
                    else:
                        V, S = self.convolution_CPU(S, V, s, w, stride, th)
                    self.layers[i]['V'][:, :, :, t] = V
                    self.layers[i]['S'][:, :, :, t] = S

                    if self.device == 'GPU':
                        S, K_inh = self.lateral_inh(S, V, K_inh, blockdim, griddim)
                    else:
                        S, K_inh = self.lateral_inh_CPU(S, V, K_inh)
                    self.layers[i]['S'][:, :, :, t] = S
                    self.layers[i]['K_inh'] = K_inh

                elif self.network_struc[i]['Type'] == 'pool':
                    if self.device == 'GPU':
                        S = self.pooling(S, s, w, stride, th, blockdim, griddim)
                    else:
                        S = self.pooling_CPU(S, s, w, stride, th)
                    self.layers[i]['S'][:, :, :, t] = S

                    if i < 3:
                        if self.device == 'GPU':
                            S, K_inh = self.lateral_inh(S, V, K_inh, blockdim, griddim)
                        else:
                            S, K_inh = self.lateral_inh_CPU(S, V, K_inh)
                        self.layers[i]['S'][:, :, :, t] = S
                        self.layers[i]['K_inh'] = K_inh
        
    @staticmethod
    @nb.jit(nopython=True)
    def pool_spikes(V,wsize):
      if wsize == -1:
        wsize_w = V.shape[0]
        wsize_h = V.shape[1]
      else:
        wsize_w=wsize
        wsize_h=wsize
      # input is a matrix with the voltage which is then pooled using a window of size wsize_w X wsize_h
      pooled_spikes = np.zeros(shape=(int(V.shape[0]/wsize_w),int(V.shape[1]/wsize_h),V.shape[2]))
      for i,j,k in np.ndindex(pooled_spikes.shape):
        pooled_spikes[i,j,k]=np.max(V[i*wsize_w:(i+1)*wsize_w,j*wsize_h:(j+1)*wsize_h,k])

      return pooled_spikes
              
    # Get training features
    def train_features(self):
        random.seed(33)
        np.random.seed(66)
        #return self.train_features_STDP()
    
        """
            Gets the train features by propagating the set of training images
            Returns:
                - X_train: Training features of size (N, M)
                            where N is the number of training samples
                            and M is the number of maps in the last layer
        """
        last_layer_shape = self.layers[self.num_layers-1]['S'].shape[:-1]
        last_layer_th = self.network_struc[5]['th']
        time_slots = self.layers[self.num_layers-1]['S'].shape[-1]
        self.class_matrix = {}
        for i in set(self.y_train):
          if self.RANDOM_INIT == 'T':
            self.class_matrix[i] = np.random.normal(0.8,0.05,last_layer_shape)
          elif self.RANDOM_INIT == 'F':
            self.class_matrix[i]=np.ones(shape=last_layer_shape) * 0.8 # np.random.normal(1,0,last_layer_shape)
          elif isinstance(self.RANDOM_INIT,float):
            self.class_matrix[i] = np.random.normal(self.RANDOM_INIT,0.05,last_layer_shape)
          
        
        self.network_struc[1]['th'] = self.TR_L1
        self.network_struc[3]['th'] = self.TR_L2 # was 50.
        self.network_struc[5]['th'] = 100000  # Set threshold of last layer to inf
        
        #MAX_VALUE = 1
        #MIN_VALUE = 0
        TR = self.TR_learning
        SPIKES_T0_CONSIDER = self.SPIKES_T0_CONSIDER
        
        train_images = list(self.spike_times_train)
        train_labels = list(self.y_train)
        
        a_plus = self.APLUS
        a_minus = self.AMINUS
        
        for iter_train in range(self.ITER_TRAIN):
        
          c = list(zip(train_images, train_labels))
          random.shuffle(c)
          train_images, train_labels = zip(*c)
        
          for img_n in range(self.num_img_train):
  
              self.reset_layers()  # Reset all layers for the new image
              if self.DoG:
                  path_img = train_images[img_n]
                  st = DoG_filter(path_img, self.filt, self.img_size, self.total_time, self.num_layers)
                  st = np.expand_dims(st, axis=2)
              else:
                  st = self.spike_times_train[img, :, :, :, :]  # (Image_number, H, W, M, time) to (H, W, M, time)
              
              label_img = train_labels[img_n]
  
              self.layers[0]['S'] = st  # (H, W, M, time)
              self.prop_step()
  
              V = self.layers[self.num_layers-1]['V']
              
              for t_slot in range(time_slots):
                
                cur_voltage = V[:,:,:,t_slot]
                if self.POOLING_W != 1:
                  cur_voltage = self.pool_spikes(cur_voltage,wsize=self.POOLING_W)
                '''
                spiking_indices = np.argwhere(cur_voltage > last_layer_th)[:5]
                if spiking_indices.shape[0] == 0:
                  order = np.argsort(-cur_voltage.ravel())[:5]
                  spiking_indices = np.stack(np.unravel_index(order,cur_voltage.shape),axis=1)
                i,j,k = spiking_indices[:,0],spiking_indices[:,1],spiking_indices[:,2]
                '''
                #take SPIKES_T0_CONSIDER top values greater than TR
                nzidx = np.where(cur_voltage >= TR)
                if SPIKES_T0_CONSIDER == -1:
                  ranking = np.argsort(cur_voltage[nzidx])[::-1]
                else:
                  ranking = np.argsort(cur_voltage[nzidx])[::-1][:SPIKES_T0_CONSIDER]
                spiking_indices = tuple(np.array(nzidx)[:, ranking])
                i,j,k = spiking_indices[0],spiking_indices[1],spiking_indices[2]
                
                n_spikes = i.size
                if n_spikes != 0:
                  for class_id in self.class_matrix:
                    if class_id == label_img:
                      self.class_matrix[label_img][i,j,k]+= a_plus* (self.class_matrix[label_img][i,j,k]*(np.ones(shape = self.class_matrix[label_img][i,j,k].shape)-self.class_matrix[label_img][i,j,k]))
                      #self.class_matrix[label_img][self.class_matrix[label_img]>MAX_VALUE]=MAX_VALUE
                      #self.class_matrix[label_img][self.class_matrix[label_img]<MIN_VALUE]=MIN_VALUE
                    else:
                      self.class_matrix[label_img][i,j,k]-= a_minus* (self.class_matrix[label_img][i,j,k]*(np.ones(shape = self.class_matrix[label_img][i,j,k].shape)-self.class_matrix[label_img][i,j,k]))
                      #self.class_matrix[label_img][self.class_matrix[label_img]>MAX_VALUE]=MAX_VALUE
                      #self.class_matrix[label_img][self.class_matrix[label_img]<MIN_VALUE]=MIN_VALUE
  

        #print('getting TRAIN score')
        correct = 0
        wrong = 0
        correct_pot = 0
        wrong_pot = 0
        classes_scores_train = []
        classes_scores_train_POT = []    
        scores_time_train = {}
            
        for img_n in range(self.num_img_train):
        
            assigned_score = assigned_pot_score = False
            
            scores_time_train[img_n]={}
            self.reset_layers()  # Reset all layers for the new image
            if self.DoG:
                path_img = next(self.train_score_img)
                st = DoG_filter(path_img, self.filt, self.img_size, self.total_time, self.num_layers)
                st = np.expand_dims(st, axis=2)
            else:
                st = self.spike_times_train[img, :, :, :, :]  # (Image_number, H, W, M, time) to (H, W, M, time)
            
            
            label_img = self.y_train[img_n]

            self.layers[0]['S'] = st  # (H, W, M, time)
            self.prop_step()

            V = self.layers[self.num_layers-1]['V']
            
            scores = {class_id:0 for class_id in set(self.y_train)}
            pot_scores = {class_id:0 for class_id in set(self.y_train)}
            
            for t_slot in range(time_slots):
              scores_time_train[img_n][t_slot]={}
              cur_voltage = V[:,:,:,t_slot]
              
              if self.POOLING_W != 1:
                cur_voltage = self.pool_spikes(cur_voltage,wsize=self.POOLING_W)
              
              nzidx = np.where(cur_voltage >= TR)
              if SPIKES_T0_CONSIDER == -1:
                ranking = np.argsort(cur_voltage[nzidx])[::-1]
              else:
                ranking = np.argsort(cur_voltage[nzidx])[::-1][:SPIKES_T0_CONSIDER]
              spiking_indices = tuple(np.array(nzidx)[:, ranking])
              i,j,k = spiking_indices[0],spiking_indices[1],spiking_indices[2]
              
              # first sum,
              for class_id in scores:
                scores[class_id]+=np.sum(self.class_matrix[class_id][i,j,k])
                pot_scores[class_id]+=np.sum(cur_voltage[i,j,k] * self.class_matrix[class_id][i,j,k])
                
                scores_time_train[img_n][t_slot][class_id] = scores[class_id]
              
              if self.CLASS_TR != -1: 
                #then check if it reached the class threshold just summing weights, and in such a casa take the max over the threshold
                if not assigned_score:
                  max_k, max_v = max(scores.items(), key=lambda t:t[1])
                  if max_v >=  self.CLASS_TR:
                    if max_k == label_img:
                      correct+=1
                    else:
                      wrong+=1
                    assigned_score = True
                  
                # also check if it reached the class threshold  summing weighted potentials, and in such a casa take the max over the threshold
                if not assigned_pot_score:
                  max_k, max_v = max(pot_scores.items(), key=lambda t:t[1])
                  if max_v >=  self.CLASS_TR:
                    if max_k == label_img:
                      correct_pot+=1
                    else:
                      wrong_pot+=1
                    assigned_pot_score = True
                if assigned_pot_score and assigned_score:
                  break
                
            scores_time_train[img_n][t_slot][class_id] = scores[class_id]
            classes_scores_train.append(tuple(scores.values()))
            classes_scores_train_POT.append(tuple(pot_scores.values()))
            
            # if there's not TR, take the max
            if self.CLASS_TR == -1: 
              if max(scores, key=scores.get) == label_img:
                correct+=1
              else:
                wrong+=1
              if max(pot_scores, key=pot_scores.get) == label_img:
                correct_pot+=1
              else:
                wrong_pot+=1
              
        train_score_str = f"train score correct: {correct}, wrong: {wrong}, p: {correct/float(self.num_img_train)}"
        train_score_str_pot = f"weighted potential train score correct: {correct_pot}, wrong: {wrong_pot}, p: {correct_pot/float(self.num_img_train)}"
        
        #print('getting TEST score')
        correct = 0
        wrong = 0
        correct_pot = 0
        wrong_pot = 0
        classes_scores_test = []
        classes_scores_test_POT = []
        scores_time_test = {}        
        
        for img_n in range(self.num_img_test):
            assigned_pot_score = False
            assigned_score = False
            scores_time_test[img_n]={}
            self.reset_layers()  # Reset all layers for the new image
            if self.DoG:
                path_img = next(self.spike_times_test)
                st = DoG_filter(path_img, self.filt, self.img_size, self.total_time, self.num_layers)
                st = np.expand_dims(st, axis=2)
            else:
                st = self.spike_times_test[img, :, :, :, :]  # (Image_number, H, W, M, time) to (H, W, M, time)
            
            label_img = self.y_test[img_n]

            self.layers[0]['S'] = st  # (H, W, M, time)
            self.prop_step()

            # Obtain maximum potential per map in last layer
            V = self.layers[self.num_layers-1]['V']
            #features = np.max(np.max(np.max(V, axis=0), axis=0), axis=1)
            
            scores = {class_id:0 for class_id in set(self.y_train)}
            pot_scores = {class_id:0 for class_id in set(self.y_train)}
            
            for t_slot in range(time_slots):
              scores_time_test[img_n][t_slot]={}
              cur_voltage = V[:,:,:,t_slot]
              
              if self.POOLING_W != 1:
                cur_voltage = self.pool_spikes(cur_voltage,wsize=self.POOLING_W)
              
              nzidx = np.where(cur_voltage >= TR)
              if SPIKES_T0_CONSIDER == -1:
                ranking = np.argsort(cur_voltage[nzidx])[::-1]
              else:
                ranking = np.argsort(cur_voltage[nzidx])[::-1][:SPIKES_T0_CONSIDER]
              spiking_indices = tuple(np.array(nzidx)[:, ranking])
              i,j,k = spiking_indices[0],spiking_indices[1],spiking_indices[2]
              
              # first sum
              for class_id in scores:
                scores[class_id]+=np.sum(self.class_matrix[class_id][i,j,k])
                pot_scores[class_id]+=np.sum(cur_voltage[i,j,k] * self.class_matrix[class_id][i,j,k])
                scores_time_test[img_n][t_slot][class_id] = scores[class_id]
              
              if self.CLASS_TR != -1: 
                #then check if it reached the class threshold just summing weights, and in such a casa take the max over the threshold
                if not assigned_score:
                  max_k, max_v = max(scores.items(), key=lambda t:t[1])
                  if max_v >=  self.CLASS_TR:
                    if max_k == label_img:
                      correct+=1
                    else:
                      wrong+=1
                    assigned_score = True
                  
                # also check if it reached the class threshold  summing weighted potentials, and in such a casa take the max over the threshold
                if not assigned_pot_score:
                  max_k, max_v = max(pot_scores.items(), key=lambda t:t[1])
                  if max_v >=  self.CLASS_TR:
                    if max_k == label_img:
                      correct_pot+=1
                    else:
                      wrong_pot+=1
                    assigned_pot_score = True
                if assigned_pot_score and assigned_score:
                  break
            
            
            classes_scores_test.append(tuple(scores.values()))
            classes_scores_test_POT.append(tuple(pot_scores.values()))
            scores_time_test[img_n][t_slot][class_id] = scores[class_id]
            
            # if there's not TR, just take the max
            if self.CLASS_TR == -1: 
              if max(scores, key=scores.get) == label_img:
                correct+=1
              else:
                wrong+=1
              if max(pot_scores, key=pot_scores.get) == label_img:
                correct_pot+=1
              else:
                wrong_pot+=1

        test_score_str = f"test score correct: {correct}, wrong: {wrong}, p: {correct/float(self.num_img_test)}"
        test_score_str_pot = f"weighted potential test score correct: {correct_pot}, wrong: {wrong_pot}, p: {correct_pot/float(self.num_img_test)}"
        
        print(f'weights_id: {self.weights_ID}, ITER_TRAIN: {self.ITER_TRAIN}, APLUS: {self.APLUS},AMINUS: {self.AMINUS},TR_learning: {self.TR_learning},SPIKES_T0_CONSIDER: {self.SPIKES_T0_CONSIDER},POOLING_W: {self.POOLING_W},RANDOM_INIT: {self.RANDOM_INIT},TR_L1: {self.TR_L1},TR_L2: {self.TR_L2},CLASS_TR: {self.CLASS_TR}\n{train_score_str} \n{test_score_str}\n{train_score_str_pot}\n{test_score_str_pot}\n')
        
        
        exit(0) 
        
  
    '''
    # Get training features
    def train_features_cp(self):
        """
            Gets the train features by propagating the set of training images
            Returns:
                - X_train: Training features of size (N, M)
                            where N is the number of training samples
                            and M is the number of maps in the last layer
        """
        
        self.network_struc[3]['th'] = 50.
        self.network_struc[5]['th'] = 100000  # Set threshold of last layer to inf
        print("-----------------------------------------------------------")
        print("----------- EXTRACTING TRAINING FEATURES ------------------")
        print("-----------------------------------------------------------")
        for i in range(self.num_img_train):
            print("------------ Train features Extraction Progress  {}%----------------".format(str(i) + '/'
                                                                                                + str(self.num_img_train)
                                                                                                + ' ('
                                                                                                + str(100 * i / self.num_img_train)
                                                                                                + ')'))

            start = timer()

            self.reset_layers()  # Reset all layers for the new image
            if self.DoG:

                path_img = next(self.spike_times_train)
                st = DoG_filter(path_img, self.filt, self.img_size, self.total_time, self.num_layers)
                st = np.expand_dims(st, axis=2)
            else:
                st = self.spike_times_train[i, :, :, :, :]  # (Image_number, H, W, M, time) to (H, W, M, time)
            self.layers[0]['S'] = st  # (H, W, M, time)
            self.prop_step()

            # Obtain maximum potential per map in last layer
            V = self.layers[self.num_layers-1]['V']
            features = np.max(np.max(np.max(V, axis=0), axis=0), axis=1)
            self.features_train.append(features)


            dt = timer() - start
            print(dt)

        # Transform features to numpy array
        n_features = self.features_train[0].shape[0]
        n_train_samples = len(self.features_train)
        X_train = np.concatenate(self.features_train).reshape((n_train_samples, n_features))
        print("------------ Train features Extraction Progress  {}%----------------".format(str(self.num_img_train)
                                                                                            + '/'
                                                                                            + str(self.num_img_train)
                                                                                            + ' ('
                                                                                            + str(100)
                                                                                            + ')'))
        print("-----------------------------------------------------------")
        print("------------- TRAINING FEATURES EXTRACTED -----------------")
        print("-----------------------------------------------------------")

        # Clear Features
        self.features_train = []
        return X_train, self.y_train
    
    
    # Get test features
    def test_features(self):
        """
            Gets the test features by propagating the set of training images
            Returns:
                - X_test: Training features of size (N, M)
                            where N is the number of training samples
                            and M is the number of maps in the last layer
        """
        self.network_struc[3]['th'] = 50.
        self.network_struc[5]['th'] = 100000  # Set threshold of last layer to inf
        print("-----------------------------------------------------------")
        print("---------------- EXTRACTING TEST FEATURES -----------------")
        print("-----------------------------------------------------------")
        for i in range(self.num_img_test):
            print("------------ Test features Extraction Progress  {}%----------------".format(str(i) + '/'
                                                                                               + str(self.num_img_test)
                                                                                               + ' ('
                                                                                               + str(100 * i / self.num_img_test)
                                                                                               + ')'))

            self.reset_layers()  # Reset all layers for the new image
            if self.DoG:
                path_img = next(self.spike_times_test)
                st = DoG_filter(path_img, self.filt, self.img_size, self.total_time, self.num_layers)
                st = np.expand_dims(st, axis=2)
            else:
                st = self.spike_times_test[i, :, :, :, :]  # (Image_number, H, W, M, time) to (H, W, M, time)
            self.layers[0]['S'] = st  # (H, W, M, time)
            self.prop_step()

            # Obtain maximum potential per map in last layer
            V = self.layers[self.num_layers-1]['V']
            features = np.max(np.max(np.max(V, axis=0), axis=0), axis=1)
            self.features_test.append(features)

        # Transform features to numpy array
        n_features = self.features_test[0].shape[0]
        n_train_samples = len(self.features_test)
        X_test = np.concatenate(self.features_test).reshape((n_train_samples, n_features))
        print("------------ Test features Extraction Progress  {}%----------------".format(str(self.num_img_test)
                                                                                           + '/'
                                                                                           + str(self.num_img_test)
                                                                                           + ' ('
                                                                                           + str(100)
                                                                                           + ')'))
        print("-----------------------------------------------------------")
        print("---------------- TEST FEATURES EXTRACTED ------------------")
        print("-----------------------------------------------------------")

        # Clear Features
        self.features_test = []
        return X_test, self.y_test
    '''
# --------------------------- CUDA interfacing functions ------------------------#
    def convolution(self, S, V, s, w, stride, th, blockdim, griddim):
        """
            Cuda Convolution Kernel call
            Returns the updated potentials and spike times
        """
        d_S = cuda.to_device(np.ascontiguousarray(S).astype(np.uint8))
        d_V = cuda.to_device(np.ascontiguousarray(V).astype(np.float32))
        d_s = cuda.to_device(np.ascontiguousarray(s).astype(np.uint8))
        d_w = cuda.to_device(np.ascontiguousarray(w).astype(np.float32))
        V_out = np.empty(d_V.shape, dtype=d_V.dtype)
        S_out = np.empty(d_S.shape, dtype=d_S.dtype)
        conv_step[griddim, blockdim](d_S, d_V, d_s, d_w, stride, th)
        d_V.copy_to_host(V_out)
        d_S.copy_to_host(S_out)
        return V_out, S_out

    def lateral_inh(self, S, V, K_inh, blockdim, griddim):
        """
            Cuda Lateral Inhibition Kernel call
            Returns the updated spike times and inhibition matrix
        """
        d_S = cuda.to_device(np.ascontiguousarray(S).astype(np.uint8))
        d_V = cuda.to_device(np.ascontiguousarray(V).astype(np.float32))
        d_K_inh = cuda.to_device(np.ascontiguousarray(K_inh).astype(np.uint8))
        S_out = np.empty(d_S.shape, dtype=d_S.dtype)
        K_inh_out = np.empty(d_K_inh.shape, dtype=d_K_inh.dtype)
        lateral_inh[griddim, blockdim](d_S, d_V, d_K_inh)
        d_S.copy_to_host(S_out)
        d_K_inh.copy_to_host(K_inh_out)
        return S_out, K_inh_out

    def pooling(self, S, s, w, stride, th, blockdim, griddim):
        """
            Cuda Pooling Kernel call
            Returns the updated spike times
        """
        d_S = cuda.to_device(np.ascontiguousarray(S).astype(np.uint8))
        d_s = cuda.to_device(np.ascontiguousarray(s).astype(np.uint8))
        d_w = cuda.to_device(np.ascontiguousarray(w).astype(np.float32))
        S_out = np.empty(d_S.shape, dtype=d_S.dtype)
        pool[griddim, blockdim](d_S, d_s, d_w, stride, th)
        d_S.copy_to_host(S_out)
        return S_out

    def STDP(self, S_sz, s, w, K_STDP, maxval, maxind1, maxind2, stride, offset, a_minus, a_plus, blockdim, griddim):
        """
            Cuda STDP-Update Kernel call
            Returns the updated weight and STDP allowed matrix
        """
        d_S_sz = cuda.to_device(np.ascontiguousarray(S_sz).astype(np.int32))
        d_s = cuda.to_device(np.ascontiguousarray(s).astype(np.uint8))
        d_w = cuda.to_device(w.astype(np.float32))
        d_K_STDP = cuda.to_device(K_STDP.astype(np.uint8))
        w_out = np.empty(d_w.shape, dtype=d_w.dtype)
        K_STDP_out = np.empty(d_K_STDP.shape, dtype=d_K_STDP.dtype)
        STDP_learning[griddim, blockdim](d_S_sz, d_s, d_w, d_K_STDP,  # Input arrays
                      maxval, maxind1, maxind2,  # Indices
                      stride, int(offset), a_minus, a_plus)  # Parameters
        d_w.copy_to_host(w_out)
        d_K_STDP.copy_to_host(K_STDP_out)
        return w_out, K_STDP_out


# --------------------------- CPU interfacing functions ------------------------#
    def convolution_CPU(self, S, V, s, w, stride, th):
        """
            CPU Convolution Function call
            Returns the updated potentials and spike times
        """
        V_out, S_out = conv_step_CPU(S, V, s, w, stride, th)
        return V_out, S_out

    def lateral_inh_CPU(self, S, V, K_inh):
        """
            CPU Lateral Inhibition Function call
            Returns the updated spike times and inhibition matrix
        """
        S_out, K_inh_out = lateral_inh_CPU(S, V, K_inh)
        return S_out, K_inh_out

    def pooling_CPU(self, S, s, w, stride, th):
        """
            CPU Pooling Function call
            Returns the updated spike times
        """
        S_out = pool_CPU(S, s, w, stride, th)
        return S_out

    def STDP_CPU(self, S_sz, s, w, K_STDP, maxval, maxind1, maxind2, stride, offset, a_minus, a_plus):
            """
                CPU STDP-Update Function call
                Returns the updated weight and STDP allowed matrix
            """
            w_out, K_STDP_out = STDP_learning_CPU(S_sz, s, w, K_STDP,  # Input arrays
                                             maxval, maxind1, maxind2,  # Indices
                                             stride, int(offset), a_minus, a_plus)  # Parameters
            return w_out, K_STDP_out
