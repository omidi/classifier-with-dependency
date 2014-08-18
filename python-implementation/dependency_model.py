
import numpy as np
import itertools
from scipy.special import gammaln

class DependecyModel:
    __pseudo_count = 0.5
    __offset = 0
    __K = 10.0   # rescaling parameter
    
    def __init__(self, featureLengthVector, zeroIndexed):
        self.featureLengthVector = featureLengthVector        
        self.numOfFeatures = len(featureLengthVector)
        self.pairFreqMatrix = np.empty((self.numOfFeatures, self.numOfFeatures), dtype=np.object)
        self.singleFreqMatrix = np.empty(self.numOfFeatures, dtype=np.object)
        self.marginalLikelihood = np.zeros(self.numOfFeatures)
        self.pseudo_count = np.zeros(self.numOfFeatures)        
        for i in xrange(self.numOfFeatures):
            self.pseudo_count[i] = 1.0 / float(featureLengthVector[i])
            self.singleFreqMatrix[i] = np.repeat(self.pseudo_count[i], featureLengthVector[i])
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            numRows, numCols = featureLengthVector[pair[0]], featureLengthVector[pair[1]]
            pseudo_count = self.pseudo_count[pair[0]] * self.pseudo_count[pair[1]]
            self.pairFreqMatrix[pair] = \
              np.repeat(pseudo_count, numRows*numCols).reshape(numRows, numCols)
        # changing the offset according to the zeroIndexed value
        if not zeroIndexed:
            self.offset = 1
        # parameters for the re-scaling logR matrix
        self.alpha = 0.
        self.beta = 0.
            

    def addToPairFreqMatrix(self, row):
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            i,j = (row[pair[0]] - self.offset), (row[pair[1]] - self.offset)
            self.pairFreqMatrix[pair][i][j] += 1.0
        for i in xrange(self.numOfFeatures):
            self.singleFreqMatrix[i][row[i] - self.offset] += 1.0
        return 0

    
    def getPairPosition(self, pair):
        if pair[0] > pair[1]: # always the first index is smaller 
            pair = pair[::-1]
        if pair[0] == pair[1]:
            return None
        return self.pairFreqMatrix[pair]

    
    def getPairFreq(self, pairFeatures, pairValues):
        if pairFeatures[0] > pairFeatures[1]: # always the first index is smaller 
            pairFeatures = pairFeatures[::-1]
        if pairFeatures[0] == pairFeatures[1]:        
            return None
        try:
            val = self.pairFreqMatrix[pairFeatures][pairValues]            
        except IndexError:
            return None
        return val

    
    def calculateLogR(self):
        self.LogR = \
          np.matrix(np.zeros(self.numOfFeatures**2).reshape(self.numOfFeatures,  \
                                                            self.numOfFeatures))
        for i in xrange(self.numOfFeatures):
            self.marginalLikelihood[i] = self.calculateMarginalLikelihood(i)
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            self.LogR[pair[::-1]] = self.LogR[pair] = self.calculateLogR_ij(pair)
            print(pair, self.LogR[pair])
            
        self.rescalingParameters()
        return 0

  
    def calculateMarginalLikelihood(self, i):
        res = .0
        N = .0
        for f in xrange(self.featureLengthVector[i]):            
            res += gammaln(self.singleFreqMatrix[i][f])
            N += self.singleFreqMatrix[i][f]
        res -= self.featureLengthVector[i]*gammaln(self.pseudo_count[i])
        res += gammaln(self.featureLengthVector[i]*self.pseudo_count[i])        
        res -= gammaln(N + self.featureLengthVector[i]*self.pseudo_count[i])
        return res 
                    
    
    def calculateLogR_ij(self, pair):
        freq = self.getPairPosition(pair)
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

    
    def updatedLogR(self, row):        
        logR_new = np.copy(self.logR)
        for pair in itertools.combinations(np.arange(self.numOfFeatures),2 ):
            i, j = pair
            freq = self.getPairPosition(pair)
            N_i, N_j = self.singleFreqMatrix[pair[0]], self.singleFreqMatrix[pair[1]]
            N = np.sum(freq)
            nrows, ncols = freq.shape
            logR_new[pair] -= gammaln(N)
            logR_new[pair] += gammaln(N + 1.0)
            logR_new[pair] -= gammaln(freq[row[i]][row[j]])
            logR_new[pair] += gammaln(freq[row[i]][row[j]] + 1.0)
            logR_new[pair] += gammaln(N_i[row[i]])
            logR_new[pair] -= gammaln(N_i[row[i]] + 1.0)
            logR_new[pair] += gammaln(N_j[row[j]])
            logR_new[pair] -= gammaln(N_j[row[j]] + 1.0)
            logR_new[pair[::-1]] = logR_new[pair]
        return logR_new

    
    def rescalingParameters(self):
        M_min, M_max = np.min(self.logR), np.max(self.logR)
        self.alpha = self.__K*np.log(10.) / (M_max - M_min)
        self.beta = -self.__K*np.log(10.) * (M_max / (M_max - M_min))
        return 0
            
    
    def rescaleMatrix(self, matrix):
        new_matrix = self.beta*np.power(np.exp(matrix), self.alpha)
        for i in xrange(matrix.shape[0]):
            new_matrix[i][i] = 0.
        return  new_matrix

    
    def laplacian(self, matrix):
        sums = np.ravel(np.sum(matrix, axis=1))
        new_matrix = -1.0*matrix
        for i in xrange(matrix.shape[0]):
            new_matrix[i][i] = sums[i]
        return new_matrix
            
        
