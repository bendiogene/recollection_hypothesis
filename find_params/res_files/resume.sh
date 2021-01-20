ITER_TRAIN="1"
RANDOM_INIT="F"
APLUS="0.01"
AMINUS="0.001"
TR_LEARNING="1"
SPK_CONSIDER="10 15 20 25 30 35 40 45 50 55 60 65 70"
POOLING_W="1"

parallel -j40 --bar --joblog /tmp/log2 python3.6 ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} >> ../find_params/res_small_try_swap.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT 