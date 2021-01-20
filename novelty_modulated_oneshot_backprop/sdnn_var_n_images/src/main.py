"""
__author__ = Nicolas Perez-Nieves
__email__ = nicolas.perez14@imperial.ac.uk

SDNN Implementation based on Kheradpisheh, S.R., et al. 'STDP-based spiking deep neural networks 
for object recognition'. arXiv:1611.01421v1 (Nov, 2016)

_Modified by Andrea Tomassilli and Zied Ben Houidi
"""

from SDNN_cuda import SDNN
from Classifier import Classifier
import numpy as np
from os.path import dirname, realpath
from math import floor
import warnings
warnings.filterwarnings('ignore')

import time
import sys

def main():

    # Flags
    learn_SDNN = False  # This flag toggles between Learning STDP and classify features
                        # or just classify by loading pretrained weights for the face/motor dataset
    if learn_SDNN:
        set_weights = False  # Loads the weights from a path (path_set_weigths) and prevents any SDNN learning
        save_weights = True  # Saves the weights in a path (path_save_weigths)
        save_features = True  # Saves the features and labels in the specified path (path_features)
    else:
        set_weights = True  # Loads the weights from a path (path_set_weigths) and prevents any SDNN learning
        save_weights = False  # Saves the weights in a path (path_save_weigths)
        save_features = False  # Saves the features and labels in the specified path (path_features)

    # ------------------------------- Learn, Train and Test paths-------------------------------#
    # Image sets directories
    path = dirname(dirname(realpath(__file__)))
    spike_times_learn = [path + '/datasets/LearningSet/Face/', path + '/datasets/LearningSet/Motor/']
    spike_times_train = [path + '/datasets/TrainingSet/Face/', path + '/datasets/TrainingSet/Motor/']
    spike_times_test = [path + '/datasets/TestingSet/Face/', path + '/datasets/TestingSet/Motor/']
    #spike_times_test = [path + '/datasets/TrainingSet/Face/', path + '/datasets/TrainingSet/Motor/']
    #spike_times_train = [path + '/datasets/TestingSet/Face/', path + '/datasets/TestingSet/Motor/']

    # Results directories
    path_set_weigths = '../find_params/sdnn_find_params/results/'
    path_save_weigths = '../find_params/sdnn_find_params/results/'
    path_features = '../find_params/sdnn_find_params/results/'

    # ------------------------------- SDNN -------------------------------#
    # SDNN_cuda parameters
    DoG_params = {'img_size': (250, 160), 'DoG_size': 7, 'std1': 1., 'std2': 2.}  # img_size is (col size, row size)
    total_time = 15
    network_params = [{'Type': 'input', 'num_filters': 1, 'pad': (0, 0), 'H_layer': DoG_params['img_size'][1],
                       'W_layer': DoG_params['img_size'][0]},
                      {'Type': 'conv', 'num_filters': 4, 'filter_size': 5, 'th': 10.},
                      {'Type': 'pool', 'num_filters': 4, 'filter_size': 7, 'th': 0., 'stride': 6},
                      {'Type': 'conv', 'num_filters': 20, 'filter_size': 17, 'th': 60.},
                      {'Type': 'pool', 'num_filters': 20, 'filter_size': 5, 'th': 0., 'stride': 5},
                      {'Type': 'conv', 'num_filters': 20, 'filter_size': 5, 'th': 2.}]

    weight_params = {'mean': 0.8, 'std': 0.01}

    max_learn_iter = [0, 3000, 0, 5000, 0, 6000, 0]
    stdp_params = {'max_learn_iter': max_learn_iter,
                   'stdp_per_layer': [0, 10, 0, 4, 0, 2],
                   'max_iter': sum(max_learn_iter),
                   'a_minus': np.array([0, .003, 0, .0003, 0, .0003], dtype=np.float32),
                   'a_plus': np.array([0, .004, 0, .0004, 0, .0004], dtype=np.float32),
                   'offset_STDP': [0, floor(network_params[1]['filter_size']),
                                   0,
                                   floor(network_params[3]['filter_size']/8),
                                   0,
                                   floor(network_params[5]['filter_size'])]}
    
    n_train_images =  int(sys.argv[2])
    # Create network
    first_net = SDNN(network_params, weight_params, stdp_params, total_time,
                     DoG_params=DoG_params, spike_times_learn=spike_times_learn,
                     spike_times_train=spike_times_train, spike_times_test=spike_times_test, device='GPU',n_train_images=n_train_images)
    

    
    first_net.weights_ID = sys.argv[1]
    first_net.w_factor=float(sys.argv[3])
    first_net.std_init=float(sys.argv[4])
    
    # set parameters
    first_net.ITER_TRAIN = 1
    first_net.APLUS = 0.007*first_net.w_factor
    first_net.AMINUS = 0.003*first_net.w_factor
    first_net.TR_learning = 10
    first_net.SPIKES_T0_CONSIDER=-1
    first_net.POOLING_W = 1
    first_net.RANDOM_INIT=0.5
    first_net.TR_L1=6.25
    first_net.TR_L2=20.75
    first_net.CLASS_TR=-1
    
    
    #print(ITER_TRAIN, APLUS, AMINUS, TR_learning, SPIKES_T0_CONSIDER, POOLING_W, RANDOM_INIT, TR_L1, TR_L2, CLASS_TR)
    
    # Set the weights or learn STDP
    weights_path = "../find_params/sdnn_original/results/"
    weight_path_list = [weights_path + 'weight_' + str(i) + '_' + first_net.weights_ID + '.npy' for i in range(len(network_params) - 1)]
    first_net.set_weights(weight_path_list)

    # Get features
    X_train, y_train = first_net.train_features()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
