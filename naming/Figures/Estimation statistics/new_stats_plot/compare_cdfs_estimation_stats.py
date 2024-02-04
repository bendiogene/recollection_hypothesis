import pandas as pd
import dabest
import pathlib

svm = [float(i) for i in pathlib.Path('./cdf_accuracies_svm_one_example.txt').read_text().splitlines()]

backprop = [float(i) for i in pathlib.Path('./cdf_accuracies_weighted.txt').read_text().splitlines()]

"""
# Load the above data into `dabest`.
iris_dabest = dabest.load(data=iris, x="species", y="petal_width",
                          idx=("setosa", "versicolor", "virginica"))
# Produce a Cumming estimation plot.
iris_dabest.mean_diff.plot()
"""
