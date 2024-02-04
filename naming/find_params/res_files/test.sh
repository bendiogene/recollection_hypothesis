ITER_TRAIN="1"
RANDOM_INIT="0.8"
APLUS="0.007"
AMINUS="0.001"
TR_LEARNING="2"
SPK_CONSIDER="-1"
POOLING_W="1"
TR_L1="1"
TR_L2="1"
CLASS_TR="2 -1"

touch ../find_params/res_tr2_p9.txt

parallel -j5 --joblog /tmp/log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT ::: $TR_L1 ::: $TR_L2 ::: $CLASS_TR 