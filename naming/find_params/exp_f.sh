ITER_TRAIN="1"
RANDOM_INIT="0.5 0.525 0.55 0.575 0.6 0.475"
APLUS="0.007 0.0071 0.0072 0.0068 0.0069"
AMINUS="0.003 0.0028 0.0029 0.0031 0.0032"
TR_LEARNING="10 11 12 13 9"
SPK_CONSIDER="-1"
POOLING_W="1"
TR_L1="6 6.25 6.5 6.75 7 5.9 6.1"
TR_L2="20 20.25 20.5 20.75 21 19.75"
CLASS_TR="-1"

touch ../find_params/res_all_spikes_same_thresholds_p4.txt

parallel -j25 --joblog /tmp/log_par.log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} >> ../find_params/res_all_spikes_same_thresholds_p4.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT ::: $TR_L1 ::: $TR_L2 ::: $CLASS_TR 
