#!/usr/bin/python3.6

# Berett Chase Babrich
# Decision Tree Classifier
# Last updated 7.16.19

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
print(feature_vals)
print(label_vals)

# create histogram matrix
hist_mat = np.zeros((2,2))
print(hist_mat)
for example in data :
	label = example[-1]
	feat_val = example[0]
	'''
	if label == 'l' :
		print('like')
	elif label == 'd' :
		print('dislike')
	'''
