
## The backpropagation-based recollection hypothesis : parameter search

Code to reproduce parameter search results. Careful! Paths to right folders need to be properly set before running the code.


### 1. Vary SDNN learning with various seeds looking for the best weights 


#### First, retrain the SDNN many times and save all resulting weights
python scripts are in: ./sdnn_original —> the input to main.py is a run_id
The shell scripts to execute experiments are: orig.sh and orig2.sh, they explore 80 run_ids all in all and output results in res_orig.txt and res_orig_2.txt

#### Second, use the saved weights    
python scripts: ./sdnn_var_weights —>  the input to main.py it’s a run_id 

### 2. Varying the weights initilization and standard deviation

We tried different weights initialization (varying init and standard deviation)

python code is in find_params/sdnn_var_init_w.py 
Shell script to launch experiments is : var_weights.sh  
it outputs results in the text file: res_w_init.txt


### 3. Further parameters we varied

- exp_f.sh, exp_f1.sh and exp_f2.sh : vary the learning rates, the thresholds, as well as various parameters and output results in : res_all_spikes_same_thresholds.txt
Use the to_table.py script to visualize results of the exploration in a human readable table. The script takes as input: res_all_spikes_same_thresholds.txt

- explore_best_sdnn_weights.sh and results in res_all_spikes_best_sdnn_weights.txt

- Vary Aplus and Aminus with Learn_SDNN set to true (relearning weights)

 

### 3. Varying number of images (Few shot learning experiments in the paper) 

pthon scripts are in: ./sdnn_var_n_images
They are invoked by the following shell script var_n_images.sh 
The script varies number of images used in one shot learning as well as weight factors and output results in : 
res_N_images_var_alpha.txt



