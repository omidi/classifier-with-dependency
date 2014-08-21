

import numpy as np
import itertools
from scipy.special import gammaln

class DependecyModel:
    __pseudo_count = 0.5
    __offset = 0
    __K = 5.0   # rescaling parameter
    
    def __init__(self, featureLengthVector, zeroIndexed):
        self.featureLengthVector = featureLengthVector        
        self.numOfFeatures = len(featureLengthVector)
        self.pairFreqMatrix = np.empty((self.numOfFeatures, self.numOfFeatures), dtype=np.object)
        self.singleFreqMatrix = np.empty(self.numOfFeatures, dtype=np.object)
        self.marginalLikelihood = np.empty(self.numOfFeatures, dtype=np.object)
        self.pseudo_count = np.zeros(self.numOfFeatures)
        # self.pseudo_count = np.repeat(self.__pseudo_count, self.numOfFeatures)        
        for i in xrange(self.numOfFeatures):
            self.pseudo_count[i] = 1.0 / float(featureLengthVector[i])
            self.singleFreqMatrix[i] = np.repeat(self.pseudo_count[i], featureLengthVector[i])
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            numRows, numCols = featureLengthVector[pair[0]], featureLengthVector[pair[1]]
            pseudo_count = self.pseudo_count[pair[0]] * self.pseudo_count[pair[1]]
            self.pairFreqMatrix[pair] = \
              np.repeat(pseudo_count, numRows*numCols).reshape(numRows, numCols)

        # The original LogR matrix that comes from the training data
        self.LogR = np.matrix(np.zeros(self.numOfFeatures**2).reshape(self.numOfFeatures, \
                                                                      self.numOfFeatures))
        # changing the offset according to the zeroIndexed value
        if not zeroIndexed:
            self.__offset = 1
        # parameter for the re-scaling logR matrix
        self.alpha = 0.
        # the determinant of the M(L(R)) matrix. NOTE: it's in log-space
        self.determinant = 0.
        # initialzing single column likelihoood
        self.singleColumnLikelihood = np.zeros(self.numOfFeatures)
        


    def finalizeModel(self):
        self.calculateLogR()
        self.rescalingParameter()
        rescaled_R = self.rescaleMatrix(self.LogR)
        self.determinant = np.log(np.linalg.det(self.laplacian(rescaled_R)[1:, 1:]))
        self.singleColumnLikelihood = np.array([self.calculateSingleColumnLikelihood(i) \
                                        for i in xrange(self.numOfFeatures)])
        # self.determinant = \
        #   np.log(np.linalg.det(self.addingIdentityMatrix(self.laplacian(rescaled_R))))
        return 0

    
    def addingIdentityMatrix(self, matrix):
        return matrix + np.identity(matrix.shape[0])

    
    def addToPairFreqMatrix(self, row):
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            i,j = (row[pair[0]] - self.__offset), (row[pair[1]] - self.__offset)
            self.pairFreqMatrix[pair][i][j] += 1.0
        for i in xrange(self.numOfFeatures):
            self.singleFreqMatrix[i][row[i] - self.__offset] += 1.0
        return 0

    
    def giveDeterminant(self):
        return self.determinant
        
    
    def givePairPosition(self, pair):
        if pair[0] > pair[1]: # always the first index is smaller 
            pair = pair[::-1]
        if pair[0] == pair[1]:
            return None
        return self.pairFreqMatrix[pair]

    
    def givePairFreq(self, pairFeatures, pairValues):
        if pairFeatures[0] > pairFeatures[1]: # always the first index is smaller 
            pairFeatures = pairFeatures[::-1]
        if pairFeatures[0] == pairFeatures[1]:        
            return None
        try:
            val = self.pairFreqMatrix[pairFeatures][pairValues]            
        except IndexError:
            return None
        return val


    def giveMarginalProbabilities(self, i):
        return self.marginalLikelihood[i]
        
    
    def calculateLogR(self):
        self.LogR = \
          np.matrix(np.zeros(self.numOfFeatures**2).reshape(self.numOfFeatures,  \
                                                            self.numOfFeatures))
        for i in xrange(self.numOfFeatures):
            self.marginalLikelihood[i] = self.calculateMarginalLikelihood(i)
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            self.LogR[pair[::-1]] = self.LogR[pair] = self.calculateLogR_ij(pair)
            print pair, self.LogR[pair]
        print('*****************')
        self.rescalingParameter()
        return 0
                    
    
    def calculateLogR_ij(self, pair):
        freq = self.givePairPosition(pair)
        N_i, N_j = self.singleFreqMatrix[pair[0]], self.singleFreqMatrix[pair[1]]
        N = np.sum(freq)
        pseudo_count = self.pseudo_count[pair[0]]*self.pseudo_count[pair[1]]        
        nrows, ncols = freq.shape
        logR = (gammaln(N) - gammaln(nrows*ncols*pseudo_count))
        logR += np.sum([gammaln(freq[ab]) for ab in \
                    itertools.product(np.arange(nrows), np.arange(ncols))])
        logR -= nrows*ncols*gammaln(pseudo_count)
        logR -= np.sum([gammaln(N_i[a]) for a in xrange(nrows)])
        logR += nrows*gammaln(self.pseudo_count[pair[0]])
        logR -= np.sum([gammaln(N_j[b]) for b in xrange(ncols)])
        logR += ncols*gammaln(self.pseudo_count[pair[1]])
        return logR

    
    def membershipTest(self, row):
        LogR_new = self.updatedLogR(row)
        rescaled_R_new = self.rescaleMatrix(LogR_new)
        dependencyPart = np.log(np.linalg.det(self.laplacian(rescaled_R_new)[1:, 1:]))
        independentPart = self.independentModel(row)
        # print dependencyPart - self.determinant
        print ('X', independentPart + dependencyPart, independentPart, dependencyPart)
        return (dependencyPart + independentPart)


    def naiveByesScore(self, row):
        score = np.sum([self.marginalLikelihood[i][row[i] - self.__offset] \
            for i in xrange(self.numOfFeatures)])
        return score


    def independentModel(self, row):
        newLikelihood = np.copy(self.singleColumnLikelihood)
        for i in xrange(self.numOfFeatures):
            newLikelihood[i] -= gammaln(self.singleFreqMatrix[i][row[i] - self.__offset])
            newLikelihood[i] += gammaln(self.singleFreqMatrix[i][row[i] - self.__offset] + 1.)
            N = np.sum(self.singleFreqMatrix[i])
            newLikelihood[i] += gammaln(N)
            newLikelihood[i] -= gammaln(N + 1.)
        return np.sum(newLikelihood)
                    
                    
    def updatedLogR(self, row):        
        logR_new = np.copy(self.LogR)
        for pair in itertools.combinations(np.arange(self.numOfFeatures),2 ):
            i, j = pair
            freq = self.givePairPosition(pair)
            N_i, N_j = self.singleFreqMatrix[pair[0]], self.singleFreqMatrix[pair[1]]
            N = np.sum(freq)
            nrows, ncols = freq.shape
            logR_new[pair] -= gammaln(N)
            logR_new[pair] += gammaln(N + 1.0)
            logR_new[pair] -= gammaln(freq[row[i] - self.__offset, row[j] - self.__offset])
            logR_new[pair] += gammaln(freq[row[i] - self.__offset, row[j] - self.__offset] + 1.0)
            logR_new[pair] += gammaln(N_i[row[i] - self.__offset])
            logR_new[pair] -= gammaln(N_i[row[i] - self.__offset] + 1.0)
            logR_new[pair] += gammaln(N_j[row[j] - self.__offset])
            logR_new[pair] -= gammaln(N_j[row[j] - self.__offset] + 1.0)
            logR_new[pair[::-1]] = logR_new[pair]
            print(pair, logR_new[pair])
        return logR_new

    
    def calculateSingleColumnLikelihood(self, i):
        res = .0
        N = np.sum(self.singleFreqMatrix[i])
        for f in xrange(self.featureLengthVector[i]):
            res += gammaln(self.singleFreqMatrix[i][f])
        res -= self.featureLengthVector[i]*self.pseudo_count[i]
        res -= gammaln(N)
        res += gammaln(self.featureLengthVector[i]*self.pseudo_count[i])
        return res
            
    
    def calculateMarginalLikelihood(self, i):
        res = np.zeros(self.featureLengthVector[i])
        N = np.log(np.sum(self.singleFreqMatrix[i]))
        for f in xrange(self.featureLengthVector[i]):
            res[f] = np.log(self.singleFreqMatrix[i][f]) - N
        return res    
    

    def rescalingParameter(self):
        M_max = np.max(self.LogR)
        if M_max < self.__K:
            M_max = 1.
            self.alpha = 1.0
            return 0
        self.alpha = self.__K / M_max
        return 0

    
    def rescaleMatrix(self, matrix):
        new_matrix = np.exp(self.alpha*matrix)
        for i in xrange(matrix.shape[0]):
            new_matrix[i,i] = 0.
        return  new_matrix

    
    def laplacian(self, matrix):
        sums = np.ravel(np.sum(matrix, axis=1))
        new_matrix = -1.0*matrix
        for i in xrange(matrix.shape[0]):
            new_matrix[i,i] = sums[i]
        return new_matrix
        
