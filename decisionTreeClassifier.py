#!/usr/bin/python3.6

# Berett Chase Babrich
# Decision Tree Classifier
# Last updated 8.7.19

# NOTES :
# A homebrewed decision tree classifier based on Daume's first chapter of "machine learning"
# one of the goals here is use as few for loops as possible, as they are gross

# imports
import re
import numpy as np
import random
import os, sys, getopt
from anytree import Node, RenderTree


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
def get_best_feat(data, feats):
    # I'm sure the time complexity could be improved somehow here
    if len(data) == 0:  # data IS empty
        if print_tables: print('data empty. choosing random label')
        guess = random.choice(label_vals)
        if print_tables: print('guess is random: ', guess)
        return -1, guess, []
    elif len(feats) == 0:  # data is NOT empty
        if print_tables: print('features empty')
        labels = list(data[:, -1])

        label_counts = [0] * len(label_vals)

        for label_unique in label_vals:
            for label in labels:
                if label == label_unique:
                    label_counts[label_vals.index(label)] += 1
        guess = None
        if all(elem == label_counts[0] for elem in label_counts):
            if print_tables: print('same number of all labels')
            guess = random.choice(label_vals)
            if print_tables: print('guess is random: ', guess)
        else:
            guess = label_vals[label_counts.index(max(label_counts))]
            if print_tables: print('most frequent label is: ', guess)
        return -1, guess, []

    # calculate feature with highest score
    # ?? could this be written entirely without for loops ??
    highest_score = 0
    f_w_highest_score = None

    # in order to check for leaf cases, we need to store all feature scores
    feat_scores = []

    for feature in range(num_features):  # last column is label, not feature
        if feature in feats:
            # create histogram matrix for current feature
            hist_mat = np.zeros((len(feature_vals[feature]), len(label_vals)))
            # populate the histogram by
            for example in data:
                label_val = example[-1]
                feat_val = example[feature]
                hist_mat[feature_vals[feature].index(feat_val)][label_vals.index(label_val)] += 1

            # sum up the highest value from each label row
            # equivalent to saying "how many would we get right
            # if we always took the most frequent value"
            argmax_arr = np.argmax(hist_mat, axis=0)
            arange_arr = np.arange(len(label_vals))
            score_numerator = np.sum(hist_mat[argmax_arr, arange_arr])

            # calculate score for current feature
            score_denominator = len(data)
            score = score_numerator / score_denominator
            feat_scores.append(score)
        else:
            feat_scores.append(-1)

    # test print
    if print_tables: print('feature scores:', feat_scores)

    # check for leaves and special case of all scores being equal
    # then return highest score
    guess = None
    if all(elem == feat_scores[0] for elem in feat_scores):  # all scores are equal
        # pick the first feature
        # this should be randomized in the future
        best_feat = random.choice(feats)

        # table printing
        if print_tables:
            print('all scores equal for this data')
            print('randomly guessing best feature as:', best_feat)

    elif all(elem == data[:, -1][0] for elem in data[:, -1]):  # all labels are equal
        # might need to be careful about doubles vs ints here?
        # note that this takes the index of the first appearance of 1.0
        guess = data[:, -1][0]
        best_feat = -1

        # table printing
        if print_tables:
            print('labels unambiguous')
            print('leaf found. Guess: ', guess)

    else:  # leaf conditions not met, take index of greatest score
        best_feat = feat_scores.index(max(feat_scores))
        if print_tables: print('best_feat', best_feat)
        feats.remove(best_feat)

    return best_feat, guess, feats


#### SPLITTING DATA
# IN: A list of lists (unsplit data), an int (feature to split data on)
# OUT: A list of lists of lists (several data lists)
def performSplitOn(data, feat):
    # table printing
    if print_tables: print('splitting on feature ', feat, '...........')

    # perform split
    # to use numpy.split, we need the indices of each feature value
    splits = []
    for f in feature_vals[feat]:
        sub_arr = [i for i in range(len(data)) if data[i][feat] == f]
        splits.append(data[sub_arr])
    return splits


#### TRAIN DECISION TREE
def trainDecisionTreeOn(data, feat_val, feats, parent_node):
    best_feat, guess, feats = get_best_feat(data, feats[:])

    # table printing
    if print_tables:
        print('in trainDecisionTree call, after get_best_feat call. Data: ')
        print(data)
        print('features left: ', feats)
        print('guess:', guess)

    # tree construction
    if guess == None:  # we are not at a leaf node
        node = Node(feat_val + ',' + str(best_feat), parent=parent_node)

        splits = performSplitOn(data, best_feat)

        for split, feat_val in zip(splits, feature_vals[best_feat]):
            child_node = trainDecisionTreeOn(split, feat_val, feats[:], node)
            child_node.parent = node

        if print_tables:
            print('all leaves taken care of')
    else:  # we are at a leaf node
        node = Node(feat_val + ',' + guess, parent=parent_node)

    return node

# assume is 2d array
def classify(data, root) :
    preds = []
    for example in data :
        next_node = root
        while next_node.name.split(',')[-1] not in label_vals:
            feat_index = next_node.name.split(',')[-1]
            val_at_feat_index = example[int(feat_index)]
            for child in next_node.children:
                if child.name[0] == val_at_feat_index:
                    next_node = child
                    break

        prediction = next_node.name.split(',')[-1]
        preds.append(prediction)

    return preds

######################################################
################## MAIN CONTROL STARTS HERE ##########
######################################################

# script parameter handling
print_tables = False
print_tree = False
try:
    opts, args = getopt.getopt(sys.argv[1:], 'h', ['print_tables', 'print_tree'])

    ''' # we will allow empty arguments for now
    if opts == [] and args == [] :
        print('empty arguments. Use -h for help')
        sys.exit(2)
    '''
except getopt.GetoptError:
    print('error with arguments passed. Use -h for help')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('version_history.py --print_tables')
        print('--print_tables shows the construction of the tree at the tabular level')
        sys.exit()
    elif opt == '--print_tables':
        print_tables = True
    elif opt == '--print_tree':
        print_tree = True

# load the file into 2d array
data_file = open('kr-vs-kp.data')
data = []
for line in data_file:
    split_line = re.split(',| |\n', line)
    if '' in split_line:
        data_line = split_line[:-1]
    else:
        data_line = split_line
    data.append(data_line)
data_file.close()
data = np.asarray(data)  # convert to numpy array

# grab label and feature values
num_features = len(data[1]) - 1
feature_vals = [[]] * num_features
label_vals = []
for example in data:
    # grab (possibly new) feature values for each feature in example
    for example_feat, feat_list in zip(example[:-1], feature_vals[:-1]):
        if example_feat not in feat_list:
            feat_list.append(example_feat)
    # grab (possibly new) label values from example
    if example[-1] not in label_vals:
        label_vals.append(example[-1])

# here we go...
feats = list(range(num_features))  # keep track of which feats have already been used
root = trainDecisionTreeOn(data, '-1', feats, Node("root"))

if print_tree:
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))


preds = classify(data, root)

print('predictions: ')
print(preds)

print('finished.')
