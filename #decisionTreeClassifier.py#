# IN: a list of categories
# OUT: the index of the category with the highest "score"
# "score" is defined by Daume in Decision trees
#def retFeatWHighestScore(features) :


# import anytree so we can "train" our decision tree
from anytree import Node, RenderTree

#the first thing we need to to is read in our data file and transform it into a 2d-array
dataFile = open('courseRatings.data','r')

dataArray = []

for line in dataFile :
    dataArray.append(line.split(','))

# note that at this point EOL chars are still present
# right now, feature f is just the first feature

features = list(range(len(dataArray[0]) - 2))

highestScore = 0
featWHighestScore = 0
print(features)

for f in features :
    categories = [] # all the different values feature f can have 
    for i in range(len(dataArray)) :
        if dataArray[i][f] not in categories :
            categories.append(dataArray[i][f])

    corGuess = 0
    
    for c in categories :
        cTot = 0
        tempCorGuess = 0
        for i in range(len(dataArray)) :
            if dataArray[i][f] == c :
                cTot += 1
                if dataArray[i][len(dataArray[0]) - 1].rstrip() == 'l' :
                    tempCorGuess += 1
        if cTot - tempCorGuess > tempCorGuess :
            tempCorGuess = cTot - tempCorGuess
        corGuess += tempCorGuess

    if corGuess/len(dataArray) > highestScore :
        highestScore = corGuess/len(dataArray)
        featWHighestScore = f
print('feat with highest score: ' + str(featWHighestScore))

# Feature with highest score has been found (call if f')

# we now create a new features list with f' removed
newFeatures = features
newFeatures.remove(featWHighestScore)
print(newFeatures)

# now we split up dataArray into subsets based on the categories of f'

categories = [] # all the different values featWHighestScore can have 
for i in range(len(dataArray)) :
    if dataArray[i][featWHighestScore] not in categories :
        categories.append(dataArray[i][featWHighestScore])

print(categories)
print(featWHighestScore)

for c in categories :
    dataArrayOfC = []
    for i in range(len(dataArray)):
        if dataArray[i][featWHighestScore] == c :
            dataArrayOfC.append(dataArray[i])
            print(dataArray[i])
    #print(dataArrayOfC)
    print('////////////////')
