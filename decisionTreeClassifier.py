# import anytree so we can "train" our decision tree
from anytree import Node, RenderTree

# RETURN BOOL FOR EXISTENCE OF DISAMBIGUITY
# IN: a data array and a list of labels
# OUT: a boolean corresponding to existence of disambiguiies
def unAmbig(dataArray, labels) :
    for l in labels :
        unAmbig = True
        for i in range(len(dataArray)) :
            if dataArray[i][len(dataArray[0]) - 1].rstrip() != l :
                unAmbig = False
        if unAmbig :
            return l
    return 'NULL'

# RETURN FEATURE WITH HIGHEST "SCORE"
# IN: features, categories, and labels (AS LISTS)
# OUT: the index of the category with the highest "score"
# -> "score" is defined by Daume in Decision trees
def featureWithHighestScore(dataArray, features, labels) :
    highestScore = 0
    featWHighestScore = 0
    for f in features :
        # grab all the different values feature f can have
        categories = []  
        for i in range(len(dataArray)) :
            if dataArray[i][f] not in categories :
                categories.append(dataArray[i][f])

        corGuess = 0

        # for each category,
        # count how many times label l appears,
        # and save the highest count
        for c in categories :
            highestNumOfCorGuesses = 0
            for l in labels :
                cTot = 0
                tempCorGuess = 0
                for i in range(len(dataArray)) :
                    if dataArray[i][f] == c :
                        cTot += 1
                        if dataArray[i][len(dataArray[0]) - 1].rstrip() == l :
                            tempCorGuess += 1

                if tempCorGuess > highestNumOfCorGuesses :
                    highestNumOfCorGuesses = tempCorGuess

            # once the highest count has been found for category c,
            # add it to a total tally 
            corGuess += highestNumOfCorGuesses

        # the total tally represents our number of correct "guesse" (Daume, p. 12)
        # this total tally divided by the number of examples gives us the score of feature f
        # save the highest score (this is what we will return)
        if corGuess/len(dataArray) > highestScore :
            highestScore = corGuess/len(dataArray)
            featWHighestScore = f
    return featWHighestScore

# TRAIN DECISION TREE
# IN: a dataArray, features, and labels, all as lists
# OUT: nothing, right now... will aid in the building of a tree
def trainDecisionTree(dataArray, features, labels) :
    if len(features) != 0 and unAmbig(dataArray, labels) == 'NULL' :

        # grab feature with highest score
        featWHighestScore = featureWithHighestScore(dataArray, features, labels)

        # Feature with highest score has been found (call if f')
        print('feat with highest score: ' + str(featWHighestScore))

        # we now create a new features list with f' removed
        newFeatures = features
        newFeatures.remove(featWHighestScore)

        # now we split up dataArray into subsets based on the categories of f'
        # and call this method on that new dataArray

        categories = [] # all the different values featWHighestScore can have 
        for i in range(len(dataArray)) :
            if dataArray[i][featWHighestScore] not in categories :
                categories.append(dataArray[i][featWHighestScore])

        for c in categories :
            dataArrayOfC = []
            for i in range(len(dataArray)):
                if dataArray[i][featWHighestScore] == c :
                    dataArrayOfC.append(dataArray[i])
                    print(dataArray[i])
            trainDecisionTree(dataArrayOfC, newFeatures, labels) # recursive call!
            print('////////////////')
    else :
        print('base case')
        print(len(features))
        print(unAmbig(dataArray, labels))
        
################## CONTROL FLOW BEGINS HERE ########################
    
#the first thing we need to to is read in our data file and transform it into a 2d-array
dataFile = open('courseRatings.data','r')

dataArray = [] # our initial data array (to be "divided and conqured" into others)

for line in dataFile :
    dataArray.append(line.split(',')) # At this point EOL chars are still present in the labels

# create features as a list of ints
features = list(range(len(dataArray[0]) - 2))

# let's grab all the possible labels a feature f can have
labels = []

for i in range(len(dataArray)) :
    if dataArray[i][len(dataArray[0]) - 1].rstrip() not in labels :
        labels.append(dataArray[i][len(dataArray[0]) - 1].rstrip())

################# TREE BUILDING STARTS HERE ########################

trainDecisionTree(dataArray, features, labels)


