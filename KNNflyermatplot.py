'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector



def labelsToColors(labels):
    # Crée un ensemble unique des étiquettes pour déterminer les classes
    uniqueLabels = unique(labels)
    # Génère des couleurs aléatoires pour chaque classe unique
    colors = plt.cm.rainbow(linspace(0, 1, len(uniqueLabels)))
    # Crée un dictionnaire pour mapper chaque étiquette à une couleur
    colorMap = {label: color for label, color in zip(uniqueLabels, colors)}
    # Mappe chaque étiquette dans l'ensemble original à une couleur
    return array([colorMap[label] for label in labels])



datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
print(datingDataMat[0 :5])
print(datingLabels[0 :5])

# Utilisation de la fonction pour convertir les étiquettes en couleurs
# Supposons que datingLabels soit déjà un tableau NumPy; sinon, convertissez-le.
datingLabelsColors = labelsToColors(array(datingLabels))

fig = plt.figure()
plt.title("Liter per Video Time")
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], c=datingLabelsColors)
plt.show()

fig = plt.figure()
plt.title("Liter per Flyer Km")
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,2], c=datingLabelsColors)
plt.show()

fig = plt.figure()
plt.title("Video TIme per Flyer Km")
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], c=datingLabelsColors)
plt.show()