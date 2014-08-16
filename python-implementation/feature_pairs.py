
import numpy as np
import itertools 

pseudo_count = 0.5

def initializePairFreqMatrix(featureLengthVector):
    numOfFeatures = len(featureLengthVector)    
    pairFreqMatrix = np.empty((numOfFeatures, numOfFeatures), dtype=np.object)
    for pair in itertools.combinations(np.arange(numOfFeatures), 2):
        numRows, numCols = featureLengthVector[pair[0]], featureLengthVector[pair[1]]
        tmp = np.repeat(pseudo_count, numRows*numCols).reshape(numRows, numCols)
        pairFreqMatrix[pair] = tmp
    return pairFreqMatrix  # 1/2 of matrix enteries are None, e.g. (2,1). Due to the fact that (1,2) is filled and identital to (2,1). Because matrix is symmetrical. It's only for the memory considerations that the (2,1) is left None.


def addToPairFreqMatrix(pairFreqMatrix, row, zeroIndexed):
    offset = 0
    if not zeroIndexed:
            offset = 1
    numOfFeatures = pairFreqMatrix.shape[0]
    for pair in itertools.combinations(np.arange(numOfFeatures), 2):
            i,j = row[pair[0]] - offset, row[pair[1]] - offset
            pairFreqMatrix[pair][i][j] += 1.0
    return 0


def createDependencyMatrix(pairFreqMatrix):
    None
    
