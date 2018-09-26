# starting over decisionTreeClassifier
# Last updated 9/26/18

# PRINT DATA MATRIX
# IN: a data array
# OUT: none. just a printing help method.
def printData(dataMat) :
    for i in range(len(dataMat)) :
        print(dataMat[i])
        
# RETURN FEATURE WITH HIGHEST "SCORE"
# IN: features, categories, and labels (AS LISTS)
# OUT: the index of the category with the highest "score"
# -> "score" is defined by Daume in Decision trees
def featureWithHighestScore(dataMat, features, labels) :
    highestScore = 0
    featWHighestScore = 0
    for f in features :
        # grab all the different values feature f can have
        categories = []  
        for i in range(len(dataMat)) :
            if dataMat[i][f] not in categories :
                categories.append(dataMat[i][f])

        corGuess = 0

        # for each category,
        # count how many times label l appears,
        # and save the highest count
        for c in categories :
            highestNumOfCorGuesses = 0
            for l in labels :
                cTot = 0
                tempCorGuess = 0
                for i in range(len(dataMat)) :
                    if dataMat[i][f] == c :
                        cTot += 1
                        if dataMat[i][len(dataMat[0]) - 1].rstrip() == l :
                            tempCorGuess += 1
                if tempCorGuess > highestNumOfCorGuesses :
                    highestNumOfCorGuesses = tempCorGuess

            # once the highest count has been found for category c,
            # add it to a total tally 
            corGuess += highestNumOfCorGuesses
        
        # the total tally represents our number of correct "guesse" (Daume, p. 12)
        # this total tally divided by the number of examples gives us the score of feature f
        # save the highest score (this is what we will return)
        if corGuess/len(dataMat) > highestScore :
            highestScore = corGuess/len(dataMat)
            featWHighestScore = f
    #print('highest score: ' + str(highestScore))
    return featWHighestScore
        
################## CONTROL FLOW BEGINS HERE ########################
    
#the first thing we need to to is read in our data file and transform it into a 2d-array
dataFile = open('courseRatings.data','r')

dataMat = [] # our initial data array (to be "divided and conqured" into others)

for line in dataFile :
    if line != '\n': # check for blank lines
        dataMat.append(line.split(',')) # At this point EOL chars are still present in the labels
        
# create features as a list of ints
features = list(range(len(dataMat[0]) - 1))

# grab all the possible labels an example can have
labels = []

for i in range(len(dataMat)) :
    if dataMat[i][len(dataMat[0]) - 1].rstrip() not in labels :
        labels.append(dataMat[i][len(dataMat[0]) - 1].rstrip()) # rstrip gets rid of those EOL chars

print(featureWithHighestScore(dataMat, features, labels))
        
dataFile.close()
