ITER_TRAIN="1"
RANDOM_INIT="0.6 0.65 0.7 0.75 0.5 0.55 0.8"
APLUS="0.007 0.006 0.0065 0.0075 0.008"
AMINUS="0.003 0.002 0.0025 0.0035"
TR_LEARNING="10 11 12 13 14"
SPK_CONSIDER="-1"
POOLING_W="1"
TR_L1="3 3.5 4 4.5 5 5.5 6 6.5 7 7.5"
TR_L2="18 18.5 19 19.5 20 20.5 21 21.5 22 22.5 23"
CLASS_TR="-1"

touch ../find_params/res_all_spikes_same_thresholds_p3.txt

parallel -j25 --joblog /tmp/log_par.log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} >> ../find_params/res_all_spikes_same_thresholds_p3.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT ::: $TR_L1 ::: $TR_L2 ::: $CLASS_TR 
