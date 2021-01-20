ITER_TRAIN="1"
RANDOM_INIT="0.8 0.5"
APLUS="0.007 0.004 0.04"
AMINUS="0 0.001 0.003 0.03"
TR_LEARNING="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
SPK_CONSIDER="-1"
POOLING_W="1"
TR_L1="10 5"
TR_L2="60 15"
CLASS_TR="-1"

touch ../find_params/res_all_spikes_same_thresholds.txt

parallel -j20 --joblog /tmp/log_par.log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} >> ../find_params/res_all_spikes_same_thresholds.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT ::: $TR_L1 ::: $TR_L2 ::: $CLASS_TR 
