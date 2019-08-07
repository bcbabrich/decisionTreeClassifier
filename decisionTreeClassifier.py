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
import random
import os, sys, getopt
import statistics
from statistics import mode

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
def get_best_feat(data, feats) :
	# TODO: guess the most prevalent answer instead of random here
	# we also do not need to return feats here
	if len(data) == 0 :
		if print_tables : print('empty data. Randomly guessing')
		guess = random.choice(label_vals)
		if print_tables : print('Guess:',guess)
		return 0, True, []
	elif len(feats) == 0 :
		if print_tables : print('no features left. Randomly guessing')
		guess = random.choice(label_vals)
		if print_tables : print('Guess:',guess)
		return 0, True, []
	
	
	# calculate feature with highest score
	# ?? could this be written entirely without for loops ??
	highest_score = 0
	f_w_highest_score = None
	
	# in order to check for leaf cases, we need to store all feature scores
	feat_scores = []
	for feature in range(num_features) : # last column is label, not feature
		if feature in feats :
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
		else : feat_scores.append(-1)

	# test print
	if print_tables : print('feature scores:',feat_scores)

	# check for leaves and special case of all scores being equal
	# then return highest score
	isLeaf = False
	if all(elem == feat_scores[0] for elem in feat_scores) : # all scores are equal
		# pick the first feature
		# this should be randomized in the future
		best_feat = 0
		
		# table printing
		if print_tables : print('all scores equal for this data')
			
	elif any(elem == 1.0 for elem in feat_scores ) : # any score is 1
		# might need to be careful about doubles vs ints here?
		# note that this takes the index of the first appearance of 1.0
		best_feat = feat_scores.index(1.0)
		isLeaf = True
		
		# table printing
		if print_tables : print('leaf found. Guess: ', label_vals[np.argmax(np.sum(hist_mat, axis=0))])
		
	else : # leaf conditions not met, take index of greatest score
		best_feat = feat_scores.index(max(feat_scores))
		if print_tables : print('best_feat',best_feat)
		feats.remove(best_feat)
	
	return best_feat, isLeaf, feats

#### SPLITTING DATA 
# IN: A list of lists (unsplit data), an int (feature to split data on)
# OUT: A list of lists of lists (several data lists)
def performSplitOn(data, feat) :
	# table printing
	if print_tables : print('splitting on feature ', feat,'...........')
		
	# perform split
	# to use numpy.split, we need the indices of each feature value
	splits = []
	for f in feature_vals[feat] :
		sub_arr = [i for i in range(len(data)) if data[i][feat] == f]
		splits.append(data[sub_arr])
	return splits

#### TRAIN DECISION TREE
# c is for counting recursive steps
def trainDecisionTreeOn(data, feats, isLeaf) :
	
	# table printing
	if print_tables :
		print('top of trainDecisionTree call. Data: ')
		print(data)
		print('features left: ', feats)
		print('isLeaf:',isLeaf)
	best_feat, isLeaf, feats = get_best_feat(data, feats[:])
	if not isLeaf :
		splits = performSplitOn(data, best_feat)
		leaves = 0
		for split in splits : 
			isLeaf = trainDecisionTreeOn(split, feats[:], False)
			
			if isLeaf : leaves += 1
			if leaves == len(splits) :
				if print_tables : print('all leaves taken care of')
				break # is this break necessary?
	return isLeaf

######################################################
################## MAIN CONTROL STARTS HERE ##########
######################################################

# script parameter handling
print_tables = False
try :
	opts, args = getopt.getopt(sys.argv[1:],'h', ['print_tables'])
	
	''' # we will allow empty arguments for now
	if opts == [] and args == [] :
		print('empty arguments. Use -h for help')
		sys.exit(2)
	'''
except getopt.GetoptError :
	print('error with arguments passed. Use -h for help')
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print 'version_history.py --print_tables'
		print('--print_tables shows the construction of the tree at the tabular level')
		sys.exit()
	elif opt == '--print_tables' :
		print_tables = True

			  
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
num_features = len(data[1]) - 1
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
feats = range(num_features) # keep track of which feats have already been used
isLeaf = trainDecisionTreeOn(data, feats, False)

print('finished.')
	
