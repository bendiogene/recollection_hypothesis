WEIGHTS_ID="0"
N_IMAGES="5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200"
FACTOR="2 3 5"
RANDOM_IN="0.05"

touch ../find_params/res_N_images_var_alpha.txt

parallel -j3 --joblog /tmp/log_par2.log python ../find_params/sdnn_var_n_images/src/main.py {1} {2} {3} {4} >> ../find_params/res_N_images_var_alpha.txt ::: $WEIGHTS_ID ::: $N_IMAGES ::: $FACTOR ::: $RANDOM_IN