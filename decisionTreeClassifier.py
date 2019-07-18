#!/usr/bin/python

# Berett Chase Babrich
# Decision Tree Classifier
# Last updated 7.18.19

# NOTES :
# A homebrewed decision tree classifier based on Daume's first chapter of "machine learning"
# one of the goals here is use as few for loops as possible, as they are gross

# imports
import re
import numpy as np

### TO-DO
# +++ read through data once to grab num_features, num_labels
#		- these are to be represented as integeres

# GET "BEST" FEATURE (feature to split on)
# IN: a list of lists (data)
# OUT: an integer (index of feature with highest score in data)
### ASSUMPTIONS MADE ABOUT DATA
# each column except the last is a feature
# every example has the same number of features
# the last column is a label
def get_best_feat(data) :
	# calculate feature with highest score
	# ?? could this be written entirely without for loops ??
	highest_score = 0
	f_w_highest_score = None
	
	# in order to check for leaf cases, we need to store all feature scores
	feat_scores = []
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
		feat_scores.append(score)
	
	# test print
	print('feature scores:',feat_scores)
	
	# check for leaves and return index of highest score if conditions not met
	isLeaf = False
	if all(elem == feat_scores[0] for elem in feat_scores) : # all scores are equal
		# pick the first feature
		# this should be randomized in the future
		best_feat = 0
		isLeaf = True
	elif any(elem == 1.0 for elem in feat_scores ) : # any score is 1
		# might need to be careful about doubles vs ints here?
		best_feat = feat_scores.index(1.0)
		isLeaf = True
	else : # leaf conditions not met, take index of greatest score
		best_feat = feat_scores.index(max(feat_scores))
	
	return best_feat, isLeaf

#### SPLITTING DATA 
# IN: A list of lists (unsplit data), an int (feature to split data on)
# OUT: A list of lists of lists (several data lists)
def performSplitOn(data, feat) :
	# perform split
	# to use numpy.split, we need the indices of each feature value
	splits = []
	for f in feature_vals[feat] :
		sub_arr = [i for i in range(len(data)) if data[i][feat] == f]
		splits.append(data[sub_arr])
	return splits

#### TRAIN DECISION TREE
# c is for counting recursive steps
def trainDecisionTreeOn(data, c) :
	print('c',c)
	c += 1
	print(data)
	while c < 10 :
		best_feat, isLeaf = get_best_feat(data)
		splits = performSplitOn(data, best_feat)
		for split in splits : 
			c = trainDecisionTreeOn(split, c) # recursive step
	return c

######################################################
################## MAIN CONTROL STARTS HERE ##########
######################################################
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
data = np.asarray(data) # convert to numpy array

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

# here we go...
trainDecisionTreeOn(data, 0)

print(splits)
