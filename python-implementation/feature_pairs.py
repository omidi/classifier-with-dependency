
import numpy as np
import itertools 

pseudo_count = 0.5

def initializePairFreqMatrix(featureLengthVector):
    numOfFeatures = len(featureLengthVector)
    pairFreqMatrix = np.empty((numOfFeatures, numOfFeatures), dtype=np.object)
    for pair in itertools.combinations(np.arange(numOfFeatures), 2):
        numRows, numCols = featureLengthVector[pair[0]], featureLengthVector[pair[1]]
        tmp = np.repeat(pseudo_count, numRows*numCols).reshape(numRows, numCols)
        # pairFreqMatrix[pair[0]][pair[1]] = tmp
        pairFreqMatrix[pair] = tmp        
        # print pair, featureLengthVector[pair[0]], featureLengthVector[pair[1]]
    return pairFreqMatrix
    
