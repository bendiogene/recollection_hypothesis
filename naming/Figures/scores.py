"""
To plot the figures like Fig3. of the paper.
"""
import matplotlib.pyplot as plt
import pickle
import os
import statistics

plt.rc('text', usetex=True)

for filename in os.listdir("./scores_classes"):
	if filename.endswith(".pkl"):
		with open(os.path.join("./scores_classes",filename),'rb') as f:
			results = pickle.load(f)
			diff_train = [(i-j) for (i,j) in results["train"]]
			c0, c1 = zip(*results["train"])
			mean_train = round(statistics.mean(c0),2),round(statistics.mean(c1),2)

			diff_test = [(i-j) for (i,j) in results["test"]]
			c0, c1 = zip(*results["test"])
			mean_test = round(statistics.mean(c0),2),round(statistics.mean(c1),2)

			fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

			plt.subplots_adjust(hspace=0.5, wspace=0.2)

			subplots = axes.flatten()

			subplots[0].set_ylabel("Face \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ Motorbike")
			subplots[0].set_xlabel("Image \#")
			subplots[0].set_title(r"\textbf{Train} "+f"(avg train = {mean_train})")
			subplots[0].scatter(list(range(len(diff_train)))[:int(len(diff_train)/2)], diff_train[:int(len(diff_train)/2)], c='b',s=17,label="Motorbike")
			subplots[0].scatter(list(range(len(diff_train)))[int(len(diff_train)/2):], diff_train[int(len(diff_train)/2):], c='r',s=17,label="Face")
			subplots[0].axhline(y=0, c='k')
			subplots[0].legend(loc='upper right')
			subplots[0].set_xlim(0,len(diff_train)+1)

			subplots[1].set_ylabel("Face \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ Motorbike")
			subplots[1].set_xlabel("Image \#")
			subplots[1].set_title(r"\textbf{Test} "+f"(avg test = {mean_test})")
			subplots[1].scatter(list(range(len(diff_test)))[:int(len(diff_test)/2)], diff_test[:int(len(diff_test)/2)], c='b',s=17)		
			subplots[1].scatter(list(range(len(diff_test)))[int(len(diff_test)/2):], diff_test[int(len(diff_test)/2):], c='r',s=17)
			subplots[1].axhline(y=0, c='k')
			subplots[1].set_xlim(0,len(diff_test)+1)

			main_title = "$it = {},\ a^+ = {},\ a^- = {},\ tr = {},\ spikes = {},\ poolw = {},\ rand = {}$".format(*os.path.splitext(filename)[0].split("_"))
			
			fig.suptitle(main_title)
			plt.savefig(os.path.join("./scores_classes",os.path.splitext(filename)[0]+".pdf"))
			plt.clf()
			print(main_title)
