ITER_TRAIN="5"
RANDOM_INIT="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
APLUS="0.007"
AMINUS="0.001"
TR_LEARNING="5"
SPK_CONSIDER="15"
POOLING_W="1"

touch .../find_params/res_init.txt

parallel -j10 --joblog /tmp/log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} >> ../find_params/res_init.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT 