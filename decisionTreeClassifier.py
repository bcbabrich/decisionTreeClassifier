#!/usr/bin/python

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
num_features = len(data[1])
feature_vals = [[]] * num_features
label_vals = []
for example in data :
	# grab (possibly new) feature values for each feature in example
	for example_feat, feat_list in zip(example[:-1], feature_vals[:-1]) :
		if example_feat not in feat_list :
			feat_list.append(example_feat)
	# grab (possibly new) label values from example
	if example[-1] not in label_vals :
		label_vals.append(example[-1])

# calculate feature with highest score
highest_score = 0
f_w_highest_score = None
for feature in range(num_features - 1) : # last column is label, not feature
	# create histogram matrix for current feature
	hist_mat = np.zeros((len(label_vals), len(feature_vals[feature])))
	
	# populate the histogram by 
	for example in data :
		label_val = example[-1]
		feat_val = example[feature]
		hist_mat[feature_vals[feature].index(feat_val)][label_vals.index(label_val)] += 1
	
	# sum up the highest value from each label row
	# equivalent to saying "how many would we get right
	# if we always took the most frequent value"
	argmax_arr = np.argmax(hist_mat, axis=1)
	arange_arr = np.arange(len(label_vals))
	score_numerator = np.sum(hist_mat[arange_arr, argmax_arr])
	
	# calculate score for current feature
	score_denominator = len(data)
	score = score_numerator / score_denominator
	print('score for feature ' + str(feature) + ': ' + str(score))
	
	# update highest score and corresponding feature
	if score > highest_score : 
		highest_score = score
		f_w_highest_score = feature

print('feature with the highest score is : ' + str(f_w_highest_score))
