from __future__ import division
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

input_f=sys.argv[1]
input_2=sys.argv[2]

x = []
y = []
x_y=[]
x_y_2=[]
with open(input_f,'r') as input:
	for line in input:
		x_y.append(float(line.split()[0]))

with open(input_2,'r') as input:
        for line in input:
                x_y_2.append(float(line.split()[0]))

sorted_data = np.sort(x_y)
yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
plt.plot(sorted_data,yvals,label='Backpropagated Action Potential')

sorted_data_2 = np.sort(x_y_2)
yvals_2=np.arange(len(sorted_data_2))/float(len(sorted_data_2)-1)
plt.plot(sorted_data_2,yvals_2,label='Machine learning (SVM)')

# plt.annotate('p-value = '+str(diff_location_ks[1]),xy=(0.005,0.6))
# plt.annotate('p-value = '+str(same_location_ks[1]),xy=(10,0.6))

plt.legend() 

plt.grid()

plt.xlabel('Accuracy')
plt.ylabel('Fraction of image pairs')
#plt.xscale('log')

plt.show()


