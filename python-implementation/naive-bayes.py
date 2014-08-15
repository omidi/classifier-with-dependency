
import csv
import operator
import numpy as np
from feature_pairs import * 

pseudo_count = 0.5

def arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Performing Naive Bayes over the data')
    parser.add_argument('-t', '--trainData',
                    action="store", dest="trainData", type=str
                    )
    parser.add_argument('-s', '--testData',
                        action="store", dest="testData", type=str
                        )
    parser.add_argument('-f', '--featureLength',
                        action="store", dest="featureLength", type=str
                        )    
    results = parser.parse_args()
    return results
    
                
def loadTrainingData(infile):
    data = {}
    with open(infile) as inf:
        for record in csv.reader(inf, delimiter='\t'):
            classId = int(record[0])
            data.setdefault( classId, [{} for f in xrange(len(record) - 1)] )
            for i in xrange(len(record) - 1):
                data[classId][i].setdefault(record[i+1], 0.)
                data[classId][i][record[i+1]] += 1.0
    return data


def probabilityValue(v, total, n):
    return (v + pseudo_count) / (total + pseudo_count*n)
    

def transferToProbabilities(data):
    probability = {}
    for classId in data:
        probability.setdefault( classId, [] )
        for feature in data[classId]:
            normalization = sum(feature.values())
            feature_length = len(feature.keys())
            prob = dict( [(key, probabilityValue(val, normalization, feature_length))
                          for key, val in feature.items()] )
            probability[classId].append(prob)
    return probability


def likelihoodOfData(data, model):
    likelihood = {}
    for classId in model.keys():
        likelihood[classId] = 0.
        for i, feature in enumerate(data):
            try:          
                likelihood[classId] += np.log(model[classId][i][feature])
            except KeyError:
                likelihood[classId] += np.log(0.000001)
    return likelihood            


def checkPrediction(real, prediction):
    return '+' if real==prediction else '-'


def parseTestData(infile, model):
    with open(infile) as inf:
        for record in csv.reader(inf, delimiter='\t'):
            likelihood = likelihoodOfData(record[1:], model)
            prediction = max(likelihood.iteritems(), key=operator.itemgetter(1))[0]
            normalization = sum( map(np.exp ,likelihood.values()) )
            print '\t'.join([
                str(prediction),
                str(np.exp(likelihood[prediction]) / normalization),
                checkPrediction(record[0], str(prediction)),
                ])

        
def trainingFeatureProbabilities(trainingMatrix, zeroIndexed, featureVector):
    offset = 0
    if not zeroIndexed:
        offset = 1
    classes = {}
    numRows, numCols = trainingMatrix.shape
    for rowIndex in xrange(numRows):
        classes.setdefault(trainingMatrix[rowIndex, 0],
                    np.array([np.repeat(pseudo_count, featureVector[x])
                    for x in xrange(len(featureVector))]))
        for colIndex in xrange(1, numCols):            
            featureIndex = trainingMatrix[rowIndex, colIndex]
            classes[trainingMatrix[rowIndex, 0]][colIndex-1][featureIndex - offset] += 1.0
    for classId in classes.keys():
        normalizationConstant = np.log(np.sum(classes[classId][0]))
        break
    for classId in classes.keys():
        for f in xrange(len(featureVector)):
            classes[classId][f] = np.log( classes[classId][f] )
        classes[classId] = classes[classId] - normalizationConstant
    return classes
                

def loadAllData(trainFile, testFile, featureLengthFile):
    zeroIndexed = False 
    with open(trainFile) as infTrain:
        with open(testFile) as infTest:   
            featureLength = len(infTrain.readline().split()) - 1
            infTrain.seek(0) # go back to the beginning of the file 
            trainData = [map(int, line.split()) for line in infTrain] 
            testData = [map(int, line.split()) for line in infTest]
    featureMatrix = np.matrix(trainData + testData, dtype=int)
    trainMatrix = np.matrix(trainData, dtype=int)
    testMatrix = np.matrix(testData, dtype=int)
    # counting the number of features for each column
    minFeature = featureMatrix[:, 1:].min(axis=0)    
    if np.any(minFeature == 0):  # not taking the class column (1-st column)
        zeroIndexed = true
    featureLengthVector = np.array([int(line.split()[-1])
                                    for line in open(featureLengthFile)])
    model = trainingFeatureProbabilities(trainMatrix, zeroIndexed, featureLengthVector) 
    return model, testMatrix, featureLengthVector, zeroIndexed


def classifiyTestSet(model, testMatrix, zeroIndexed):
    offset = 0
    if not zeroIndexed:
        offset = 1
    numRows, numCols = testMatrix.shape
    predictions = []
    for matrixRow in testMatrix:
        row = np.ravel(matrixRow)
        likelihood = {}
        for classId in model.keys():
            likelihood[classId] = 0.
            for col in xrange(1, numCols):
                likelihood[classId] += model[classId][col-1][row[col]-offset]
        bestPrediction = max(likelihood.iteritems(), key=operator.itemgetter(1))[0]
        normalization = sum( map(np.exp, likelihood.values()) )
        predictions.append( (
            row[0], # the 'true' value for class
            bestPrediction,   # the predicted class 
            np.exp(likelihood[bestPrediction]) / normalization, # the posterior to belong to class
            ) )
    return predictions


def performanceCheck(predictions):
    correctnessTest = lambda p: 1 if p[0]==p[1] else 0
    return np.sum([correctnessTest(p) for p in predictions])

    
def main():
    args = arguments()
    model, testMatrix, featureLengthVector, zeroIndexed = \
        loadAllData(args.trainData, args.testData, args.featureLength)
    print initializePairFreqMatrix(featureLengthVector)
    exit()        
    predictions = classifiyTestSet(model, testMatrix, zeroIndexed)
    test = lambda p: '+' if p[0]==p[1] else '-'
    for pred in predictions:
        print '\t'.join([
            str(pred[0]),
            str(pred[1]),
            str(pred[2]),
            test(pred),
            ])
    print performanceCheck(predictions)    
    
if __name__ == '__main__':
    main()
    
