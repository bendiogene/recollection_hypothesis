awk '$1=="Train"{i+=1;sum+=$5}END{print i,sum/i}' ./multiple_svms_oneimage_2.txt
awk '$1=="Train"{print $5}' ./multiple_svms_oneimage_2.txt > cdf_svm_results_one_images_2.txt
awk '$1=="Train"{i+=1;sum+=$5}END{print i,sum/i}' ./multiple_svms_oneimage.txt
awk '$1=="Train"{print $5}' /home/zbenhoui/tmp/SDNN_python/multiple_svms_oneimage.txt > cdf_svm_results_one_images.txt