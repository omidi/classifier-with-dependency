

import numpy as np
import itertools
from scipy.special import gammaln

class DependecyModel:
    __pseudo_count = 0.5
    __K = 10.   # default value for the rescaling parameter
    
    def __init__(self, featureLengthVector):
        self.featureLengthVector = featureLengthVector        
        self.numOfFeatures = len(featureLengthVector)
        self.pairFreqMatrix = np.empty((self.numOfFeatures, self.numOfFeatures), dtype=np.object)
        self.singleFreqMatrix = np.empty(self.numOfFeatures, dtype=np.object)
        self.marginalLikelihood = np.empty(self.numOfFeatures, dtype=np.object)
        self.pseudo_count = np.zeros(self.numOfFeatures)
        # self.pseudo_count = np.repeat(self.__pseudo_count, self.numOfFeatures)        
        for i in xrange(self.numOfFeatures):
            self.pseudo_count[i] = 1.0 / float(featureLengthVector[i])
            self.singleFreqMatrix[i] = np.repeat(self.pseudo_count[i], featureLengthVector[i]+1) # one extra entery for missing data
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            i,j = pair
            numRows, numCols = featureLengthVector[i]+1, featureLengthVector[j]+1 # extra row and column for missing data
            pseudo_count = self.pseudo_count[i] * self.pseudo_count[j]
            self.pairFreqMatrix[pair] = \
              np.repeat(pseudo_count, numRows*numCols).reshape(numRows, numCols)
        # The original LogR matrix that comes from the training data
        self.LogR = np.matrix(np.zeros(self.numOfFeatures**2).reshape(self.numOfFeatures, \
                                                                      self.numOfFeatures))
        # parameter for the re-scaling logR matrix
        self.alpha = 0.
        # the determinant of the M(L(R)) matrix. NOTE: it's in log-space
        self.determinant = 0.
        # initialzing single column likelihoood
        self.singleColumnLikelihood = np.zeros(self.numOfFeatures)
        self.independentLikelihood = 0.
        


    def finalizeModel(self):
        self.calculateLogR()
        self.rescalingParameter()
        rescaled_R = self.rescaleMatrix(self.LogR)
        # self.determinant = np.log(np.linalg.det(self.laplacian(rescaled_R)[1:, 1:]))
        self.determinant = np.log(np.linalg.det(self.laplacian(rescaled_R) + \
                                                np.identity(self.numOfFeatures)))
        self.singleColumnLikelihood = np.array([self.calculateSingleColumnLikelihood(i) \
                                        for i in xrange(self.numOfFeatures)])
        self.independentLikelihood = np.sum(self.singleColumnLikelihood)
        # self.determinant = \
        #   np.log(np.linalg.det(self.addingIdentityMatrix(self.laplacian(rescaled_R))))
        return 0


    def changeRescalingParams(self, new_K):
        self.__K = float(new_K)
        self.rescalingParameter()
        rescaled_R = self.rescaleMatrix(self.LogR)
        # self.determinant = np.log(np.linalg.det(self.laplacian(rescaled_R)[1:, 1:]))
        self.determinant = np.log(np.linalg.det(self.laplacian(rescaled_R) + \
                                                np.identity(self.numOfFeatures)))                
        # print(self.determinant, self.__K, self.alpha)
        return 0


    def fittingRho(self):
        rescaled_R = self.rescaleMatrix(self.LogR)        
        rho = 0.01
        for i in xrange(100):
            determinant = np.log(np.linalg.det(self.laplacian(rescaled_R) + \
                                                    rho*np.identity(self.numOfFeatures)))
            print rho,'\t', determinant
            rho += 0.01
        return 0
            
            

    def addingIdentityMatrix(self, matrix):
        return matrix + np.identity(matrix.shape[0])

    
    def addToPairFreqMatrix(self, row):
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            i,j = pair[0], pair[1]            
            if row[i] > 0 and row[j] > 0:
                self.pairFreqMatrix[pair][row[i]][row[j]] += 1.0
            elif row[i]==0 and row[j]>0:
                weight_i = 1.0 / self.featureLengthVector[i]
                for f_i in xrange(1, self.featureLengthVector[i]+1):
                    self.pairFreqMatrix[pair][f_i][row[j]] += weight_i
            elif row[i]>0 and row[j]==0:
                weight_j = 1.0 / self.featureLengthVector[j]
                for f_j in xrange(1, self.featureLengthVector[j]+1):
                    self.pairFreqMatrix[pair][row[i]][f_j] += weight_j
            else:
                weight = 1.0 / (self.featureLengthVector[i]*self.featureLengthVector[j])
                for f_i in xrange(1, self.featureLengthVector[i]+1):
                    for f_j in xrange(1, self.featureLengthVector[j]+1):
                        self.pairFreqMatrix[pair][f_i][f_j] += weight        
        for i in xrange(self.numOfFeatures):
            if row[i]>0:
                self.singleFreqMatrix[i][row[i]] += 1.0
            else:
                weight = 1.0 / self.featureLengthVector[i]
                for f in xrange(1, self.featureLengthVector[i]+1):
                    self.singleFreqMatrix[i][f] += weight
        return 0

    
    def giveDeterminant(self):
        return self.determinant
        
    
    def givePairPosition(self, pair):
        i,j = pair
        if i > j: # always the first index is smaller 
            pair = pair[::-1]
        if i == j:
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
        # for i in xrange(self.numOfFeatures):
        #     self.marginalLikelihood[i] = self.calculateMarginalLikelihood(i)
        for pair in itertools.combinations(np.arange(self.numOfFeatures), 2):
            self.LogR[pair[::-1]] = self.LogR[pair] = self.calculateLogR_ij(pair)
        self.rescalingParameter()
        return 0
                    
    
    def calculateLogR_ij(self, pair):
        freq = self.givePairPosition(pair)
        i,j = pair
        N_i, N_j = self.singleFreqMatrix[i], self.singleFreqMatrix[j]
        N = np.sum(freq[1:,1:]) # because first row and column are for missing data
        pseudo_count = self.pseudo_count[i]*self.pseudo_count[j]
        nrows, ncols = freq.shape[0]-1, freq.shape[1]-1
        logR = (gammaln(N) - gammaln(nrows*ncols*pseudo_count))
        logR += np.sum([gammaln(freq[ab]) for ab in \
                    itertools.product(np.arange(1, nrows+1), np.arange(1, ncols+1))])
        logR -= nrows*ncols*gammaln(pseudo_count)
        logR -= np.sum([gammaln(N_i[a]) for a in xrange(1, nrows+1)])
        logR += nrows*gammaln(self.pseudo_count[i])
        logR -= np.sum([gammaln(N_j[b]) for b in xrange(1, ncols+1)])
        logR += ncols*gammaln(self.pseudo_count[j])
        return logR

    
    def membershipTest(self, row):
        LogR_new = self.updatedLogR(row)
        rescaled_R_new = self.rescaleMatrix(LogR_new)
        dependencyPart = np.log(np.linalg.det(self.laplacian(rescaled_R_new) + \
                                                np.identity(self.numOfFeatures))) - self.determinant
        # dependencyPart = np.log(np.linalg.det(self.laplacian(rescaled_R_new)[1:, 1:])) - self.determinant
        independentPart = self.independentModel(row) 
        return (dependencyPart + independentPart)


    def naiveBayesScore(self, row):
        score = np.sum([self.marginalLikelihood[i][row[i] - self.__offset] \
            for i in xrange(self.numOfFeatures)])
        return score


    def independentModel(self, row):
        newLikelihood = np.copy(self.singleColumnLikelihood)        
        for i in xrange(self.numOfFeatures):
            N = np.sum(self.singleFreqMatrix[i][1:])
            newLikelihood[i] += gammaln(N)
            newLikelihood[i] -= gammaln(N + 1.)            
            if row[i]>0:
                newLikelihood[i] -= gammaln(self.singleFreqMatrix[i][row[i]])
                newLikelihood[i] += gammaln(self.singleFreqMatrix[i][row[i]] + 1.)
            else:
                weight = 1.0 / self.featureLengthVector[i]
                for f in xrange(1, self.featureLengthVector[i]+1):
                    newLikelihood[i] -= gammaln(self.singleFreqMatrix[i][f])
                    newLikelihood[i] += gammaln(self.singleFreqMatrix[i][f] + weight)
        return (np.sum(newLikelihood) - self.independentLikelihood)
                    
                    
    def updatedLogR(self, row):        
        logR_new = np.copy(self.LogR)
        for pair in itertools.combinations(np.arange(self.numOfFeatures),2 ):
            i, j = pair
            freq = self.givePairPosition(pair)
            N_i, N_j = self.singleFreqMatrix[i], self.singleFreqMatrix[j]
            N = np.sum(freq[1:,1:])
            nrows, ncols = (freq.shape[0]-1), (freq.shape[1]-1)
            logR_new[pair] -= gammaln(N)
            logR_new[pair] += gammaln(N + 1.0)
            if row[i]>0 and row[j]>0:
                logR_new[pair] -= gammaln( freq[row[i],row[j]] )
                logR_new[pair] += gammaln( freq[row[i],row[j]] + 1.0 )
                logR_new[pair] += gammaln( N_i[row[i]] )
                logR_new[pair] -= gammaln( N_i[row[i]] + 1.0 )
                logR_new[pair] += gammaln( N_j[row[j]] )
                logR_new[pair] -= gammaln( N_j[row[j]] + 1.0 )
            elif row[i]==0 and row[j]>0:
                weight_i = 1.0 / self.featureLengthVector[i]
                for f_i in xrange(1, self.featureLengthVector[i]+1):
                    logR_new[pair] -= gammaln( freq[f_i, row[j]] )
                    logR_new[pair] += gammaln( freq[f_i, row[j]] + weight_i )
                    logR_new[pair] += gammaln( N_i[f_i] )
                    logR_new[pair] -= gammaln( N_i[f_i] + weight_i )
                logR_new[pair] += gammaln( N_j[row[j]] )
                logR_new[pair] -= gammaln( N_j[row[j]] + 1.0 )
            elif row[i]>0 and row[j]==0:
                weight_j = 1.0 / self.featureLengthVector[j]
                for f_j in xrange(1, self.featureLengthVector[j]+1):
                    logR_new[pair] -= gammaln( freq[row[i], f_j] )
                    logR_new[pair] += gammaln( freq[row[i], f_j] + weight_j )
                    logR_new[pair] += gammaln( N_j[f_j] )
                    logR_new[pair] -= gammaln( N_j[f_j] + weight_j )
                logR_new[pair] += gammaln( N_i[row[i]] )
                logR_new[pair] -= gammaln( N_i[row[i]] + 1.0 )
            else:
                weight = 1.0 / (self.featureLengthVector[i]*self.featureLengthVector[j])
                weight_j = 1.0 / self.featureLengthVector[j]
                weight_i = 1.0 / self.featureLengthVector[i]
                for f_i in xrange(1, self.featureLengthVector[i] + 1):
                    logR_new[pair] += gammaln(N_i[f_i])
                    logR_new[pair] -= gammaln(N_i[f_i] + weight_i)                    
                    for f_j in xrange(1, self.featureLengthVector[j]+1):
                        logR_new[pair] -= gammaln(freq[f_i, f_j])
                        logR_new[pair] += gammaln(freq[f_i, f_j] + weight)
                for f_j in xrange(1, self.featureLengthVector[j]+1):
                    logR_new[pair] += gammaln(N_j[f_j])
                    logR_new[pair] -= gammaln(N_j[f_j] + weight_j)                    
                    
            logR_new[pair[::-1]] = logR_new[pair]                
        return logR_new

    
    def calculateSingleColumnLikelihood(self, i):
        res = .0
        N = np.sum(self.singleFreqMatrix[i][1:])
        for f in xrange(1, self.featureLengthVector[i]+1):
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
