cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 10.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 20.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 30.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 40.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 50.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 55.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 60.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 65.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 70.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 85.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'
cat res_novelty_habits_vary_factor_bestsdnn.txt | grep  "w_factor 90.0" | awk '{sum+=$10;i+=1}$10>max{max=$10}END{print $6,sum/i,max}'


