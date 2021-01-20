RUN_ID="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"

touch ../find_params/res_orig.txt

parallel -j10 --joblog /tmp/log_par2.log python ../find_params/sdnn_original/src/main.py {1} >> ../find_params/res_orig.txt ::: $RUN_ID
