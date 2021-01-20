WEIGHTS_ID="61"
N_IMAGES="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
FACTOR="10 20 30 40 50 55 60 65 70 85 90"
RANDOM_IN="0.05"

touch ./res_novelty_habits_vary_factor_bestsdnn.txt

parallel -j10 --joblog /tmp/log_parzi_varyfactor_bestsdnn.log python ./sdnn_var_n_images/src/main_vary_images_best_sdnn.py {1} {2} {3} {4} >> ./res_novelty_habits_vary_factor_bestsdnn.txt ::: $WEIGHTS_ID ::: $N_IMAGES ::: $FACTOR ::: $RANDOM_IN
