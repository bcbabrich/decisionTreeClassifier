#!/usr/bin/python3.6

# Berett Chase Babrich
# Decision Tree Classifier
# Last updated 7.17.19

# imports
import re
import numpy as np

### TO-DO
# +++ read through data once to grab num_features, num_labels
#		- these are to be represented as integeres

### ASSUMPTIONS MADE ABOUT DATA
# each column except the last is a feature
# every example has the same number of features
# the last column is a label

# load the file into 2d array
data_file = open('courseRatings.data')
data = []
for line in data_file :
	split_line = re.split(',| |\n', line)
	if '' in split_line :
		data_line = split_line[:-1]
	else :
		data_line = split_line
	data.append(data_line)
data_file.close()

# grab label and feature values
feature_vals = [[]] * len(data[1])
label_vals = []
for example in data :
	# grab (possibly new) feature values for each feature in example
	for example_feat, feat_list in zip(example[:-1], feature_vals[:-1]) :
		if example_feat not in feat_list :
			feat_list.append(example_feat)
	# grab (possibly new) label values from example
	if example[-1] not in label_vals :
		label_vals.append(example[-1])

# create histogram matrix
# the '0' will eventually be replaced with an iterator
hist_mat = np.zeros((len(label_vals), len(feature_vals[0])))
for example in data :
	label_val = example[-1]
	feat_val = example[0]
	hist_mat[feature_vals[0].index(feat_val)][label_vals.index(label_val)] += 1
print(hist_mat)
print('numpy array of argmax',np.argmax(hist_mat, axis=1))
argmax_arr = np.argmax(hist_mat, axis=1)
print('arange',np.arange(hist_mat.shape[0]))
arange_arr = np.arange(hist_mat.shape[0])
print(hist_mat[arange_arr, argmax_arr])
print(np.sum(hist_mat[arange_arr, argmax_arr]))
print('------')

test_mat = np.array([[1,5,2],[7,1,3],[8,9,10]])
print(test_mat)
print('numpy array of argmax',np.argmax(test_mat, axis=1))
argmax_arr = np.argmax(test_mat, axis=1)
print('arange',np.arange(test_mat.shape[0]))
arange_arr = np.arange(test_mat.shape[0])
print(np.sum(test_mat[arange_arr, argmax_arr]))
