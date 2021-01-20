## One shot learning experiments

In these experiments, co-occurence learning is done in a one-shot fashion: 
The neural networks are shown only a single example of each class: an example of moto with the correct label and an example of face with the correct label.
The test is then done on the entire test dataset.
 
In these experiments, we vary the pairs of images that we use for the supervised co-occurence learning step. 
We find that results curiously vary depending on which pair is used to learn.

We go through the steps involved through the paper experimennts.

### 1. Assess the impact of learning rate on one-shot experiments:

First thing we do is to pick the learning rate value for this co-occurence phase. 
One shot experiments need a high learning rate so as to remember fast in one single time. 
Many learning rate increasing factors have thus been tried. Steps involved are as follows.

#### 1.1. Vary learning rates with shell script: guess_from_one_bestsdnn_vary_factor.sh

The script in the example varies the lambda multiplier of the learning rate within this range (10 20 30 40 50 55 60 65 70 85 90)
This outputs the file "res_novelty_habits_vary_factor_bestsdnn.txt"

In reality, for the paper, we tested all the following parameters in addition: 
60 61 62 63 64 65 66 67 68 69 70 72 75 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 150 200 87.5 87.6 87.7 87.8 87.9 88 88.1 88.2 88.3 88.4 88.5 89.9 90 90.1 140 149 151 160 190 200 500 1000 15000 

#### 1.2. Analyze performance results with the following script: print_factor_mean_perf.sh 

The script takes as input the previous output: "res_novelty_habits_vary_factor_bestsdnn.txt" 
It outputs for each Lambda in the paper (w_factor here) the mean accuracy across pairs of images. 
Example of output file for all our parameters: results_vary_factor.txt

From the results, one can see that all lambdas from 60 to 200 give overall decent results with little variations. 
Extremely high learning rates (e.g. from 500 on) result in no learning at all.  

For the paper, we pick one of the decent values (a lambda of 65), which is not necessarily the best. 
After all, our goal is to assess plausibility, not to achieve the best accuracy. 

### 2.Finally run the oneshot experiments for Backpropagated AP with : var_guess_from_onesingle_bestsdnn.sh 

The script generates an output similar to : res_novelty_habits_vary_factor_and_images_learnfrom_onesingle_bestsdnn.txt

In the paper we run the script multiple times, varying each time the pairs of images. A more exhaustive list of experiment results can be found in: res_novelty_habits_learn_from_oneimage.txt

A file like the latter is used to produce the CDFs of accuracies...

