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

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector(f'trainingDigits/{fileNameStr}')
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(f'testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(f"the classifier came back with: {classifierResult} , the real answer is: {classNumStr}")
        if (classifierResult != classNumStr): errorCount += 1.0
    print(f"\nthe total number of errors is: {errorCount}")
    print(f"\nthe total error rate is: {(errorCount/float(mTest))}")

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1)) 
    return normDataSet, ranges, minVals

handwritingClassTest()
"""
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
"""