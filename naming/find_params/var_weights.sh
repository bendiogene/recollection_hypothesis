WEIGHTS_INIT='0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1'
STD_IN="0 0.05 0.1"

touch ../find_params/res_w_init.txt

parallel -j3 --joblog /tmp/log_par2.log python ../find_params/sdnn_var_init_w/src/main.py {1} {2} {3} {4} >> ../find_params/res_w_init.txt ::: $WEIGHTS_INIT ::: $STD_IN