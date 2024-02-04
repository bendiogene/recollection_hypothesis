ITER_TRAIN="1 10 25 50"
RANDOM_INIT="F T"
APLUS="0.002 0.003 0.004 0.007 0.01 0.025 0.05 0.1 0.25 0.5"
AMINUS="0 0.002 0.003 0.004 0.007 0.01 0.025 0.05 0.1 0.25 0.5"
TR_LEARNING="1 3 5 10"
SPK_CONSIDER="1 5 10 15 -1"
POOLING_W="1 2 3"

touch ../find_params/res.txt

parallel -j40 --joblog /tmp/log python3.6 ../find_params/sdnn_find_params/src/main.py {1} {2} {3} {4} {5} {6} {7} >> ../find_params/res.txt ::: $ITER_TRAIN ::: $APLUS ::: $AMINUS ::: $TR_LEARNING ::: $SPK_CONSIDER ::: $POOLING_W ::: $RANDOM_INIT 