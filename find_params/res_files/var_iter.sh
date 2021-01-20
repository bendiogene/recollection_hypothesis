ITER_TRAIN="1 5 10 20 30 40 50 100 150 200 500"
RANDOM_INIT="0.8"
APLUS="0.007"
AMINUS="0.001"
TR_LEARNING="5"
SPK_CONSIDER="15"
POOLING_W="1"

touch ../find_params/res_iters.txt

parallel -j10 --joblog /tmp/log python ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} >> ../find_params/res_iters.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT 