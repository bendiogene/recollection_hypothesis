ITER_TRAIN="1"
RANDOM_INIT="0.6 0.7 0.5 0.8"
APLUS="0.007 0.004"
AMINUS="0.003"
TR_LEARNING="8 9 10 11 12 13"
SPK_CONSIDER="-1"
POOLING_W="1"
TR_L1="20 15 10 5"
TR_L2="10 15 20 25"
CLASS_TR="-1"

touch ../find_params/res_all_spikes_same_thresholds_p2.txt

parallel -j20 --joblog /tmp/log_par.log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} >> ../find_params/res_all_spikes_same_thresholds_p2.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT ::: $TR_L1 ::: $TR_L2 ::: $CLASS_TR 
