ITER_TRAIN="1"
RANDOM_INIT="0.5 0.55 0.6"
APLUS="0.007"
AMINUS="0.003"
TR_LEARNING="10 11 12 13 9 8 7 19 20 21 22 23 30 31 32 40 41 42 50 51 52"
SPK_CONSIDER="-1"
POOLING_W="1"
TR_L1="10 10.25 10.5 10.75 9 9.9 9.1 11"
TR_L2="60 60.25 60.5 60.75 59 59.75 58 57 61"
CLASS_TR="-1"

touch ../find_params/res_all_spikes_best_sdnn_weights_10_60.txt

parallel -j25 --joblog /tmp/log_par.log python ../find_params/find_params_best_weights/src/main.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} >> ../find_params/res_all_spikes_best_sdnn_weights_10_60.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT ::: $TR_L1 ::: $TR_L2 ::: $CLASS_TR 
