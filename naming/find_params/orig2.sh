RUN_ID="41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80"

touch ../find_params/res_orig_2.txt

parallel -j10 --joblog /tmp/log_par2.log python ../find_params/sdnn_original/src/main.py {1} >> ../find_params/res_orig_2.txt ::: $RUN_ID
