

import random
import operator

class Column:
    def __init__(self, inputColumn, rule):
        self.rule = rule
        self.column = self.map(inputColumn)    
        
    def map(self, column):
        resColumn = []
        for v in column:
            dice = random.random()
            cumulativeProbability = 0.
            decisionRules = sorted(self.rule[v].iteritems(), key=operator.itemgetter(1))
            for decision, probability in decisionRules:
                cumulativeProbability += probability
                if dice <= cumulativeProbability:
                    resColumn.append(decision)
                    break
        return resColumn

    def giveColumn(self):
        return self.column
                    

def uniformColumn(length, featureLength):        
    return [random.randrange(1,featureLength+1) for i in xrange(length)]


def nondependencyData():
    """
    """
    numOfDataPoints = 1000
    rootColumn = uniformColumn(numOfDataPoints, 4)
    columnNum2 = uniformColumn(numOfDataPoints, 4)
    columnNum3 = uniformColumn(numOfDataPoints, 4)
    columnNum4 = uniformColumn(numOfDataPoints, 4)
    columnNum5 = uniformColumn(numOfDataPoints, 4)
    columnNum6 = uniformColumn(numOfDataPoints, 4)
    columnNum7= uniformColumn(numOfDataPoints, 4)
    columnNum8= uniformColumn(numOfDataPoints, 4)
    for i in xrange(numOfDataPoints):        
        print '\t'.join(map(str, [
            '1',
        rootColumn[i],
        columnNum2[i], 
        columnNum3[i],
        columnNum4[i],
        columnNum5[i],
        columnNum6[i],
        columnNum7[i],
        columnNum8[i],
        ]))
    rootColumn = uniformColumn(numOfDataPoints, 4)
    columnNum2 = uniformColumn(numOfDataPoints, 4)
    columnNum3 = uniformColumn(numOfDataPoints, 4)
    columnNum4 = uniformColumn(numOfDataPoints, 4)
    columnNum5 = uniformColumn(numOfDataPoints, 4)
    columnNum6 = uniformColumn(numOfDataPoints, 4)
    columnNum7= uniformColumn(numOfDataPoints, 4)
    columnNum8= uniformColumn(numOfDataPoints, 4)
    for i in xrange(numOfDataPoints):        
        print '\t'.join(map(str, [
            '2',
        rootColumn[i],
        columnNum2[i], 
        columnNum3[i],
        columnNum4[i],
        columnNum5[i],
        columnNum6[i],
        columnNum7[i],
        columnNum8[i],
        ]))
    return 0


def produceDependencyData():
    numOfDataPoints = 1000
    rootColumn = uniformColumn(numOfDataPoints, 4)
    ###
    columnNum2Rule = {
        1:{1:.7, 2:.1, 3:.1, 4:.1},
        2:{1:.1, 2:.7, 3:.1, 4:.1},
        3:{1:.1, 2:.1, 3:.7, 4:.1},
        4:{1:.1, 2:.1, 3:.1, 4:.7},        
        }        
    columnNum2Object = Column(rootColumn, columnNum2Rule)
    columnNum2 = columnNum2Object.giveColumn()
    ###    
    columnNum3Rule = {
        1:{1:.1, 2:.7, 3:.1, 4:.1},
        2:{1:.1, 2:.1, 3:.7, 4:.1},
        3:{1:.7, 2:.1, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.1, 4:.7},        
        }        
    columnNum3Object = Column(columnNum2, columnNum3Rule)
    columnNum3 = columnNum3Object.giveColumn()
    ###    
    columnNum4Rule = {
        1:{1:.1, 2:.1, 3:.7, 4:.1},
        2:{1:.1, 2:.7, 3:.1, 4:.1},
        3:{1:.1, 2:.1, 3:.1, 4:.7},
        4:{1:.7, 2:.1, 3:.1, 4:.1},        
        }        
    columnNum4Object = Column(columnNum3, columnNum4Rule)
    columnNum4 = columnNum4Object.giveColumn()
    ###
    columnNum5Rule = {
        1:{1:.1, 2:.1, 3:.7, 4:.1},
        2:{1:.7, 2:.1, 3:.1, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.1, 4:.7},        
        }        
    columnNum5Object = Column(columnNum4, columnNum5Rule)
    columnNum5 = columnNum5Object.giveColumn()
    ###
    columnNum6Rule = {
        1:{1:.7, 2:.1, 3:.1, 4:.1},
        2:{1:.1, 2:.1, 3:.7, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.1, 4:.7},        
        }        
    columnNum6Object = Column(columnNum5, columnNum6Rule)
    columnNum6 = columnNum6Object.giveColumn()
    ###
    columnNum7Rule = {
        1:{1:.1, 2:.1, 3:.1, 4:.7},
        2:{1:.7, 2:.1, 3:.7, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.7, 4:.1},        
        }        
    columnNum7Object = Column(columnNum6, columnNum7Rule)
    columnNum7 = columnNum7Object.giveColumn()
    random.shuffle(columnNum7)
    ###
    columnNum8Rule = {
        1:{1:.03, 2:.03, 3:.03, 4:.91},
        2:{1:.03, 2:.91, 3:.03, 4:.03},
        3:{1:.03, 2:.03, 3:.91, 4:.03},
        4:{1:.91, 2:.03, 3:.03, 4:.03},        
        }        
    columnNum8Object = Column(columnNum7, columnNum8Rule)
    columnNum8 = columnNum8Object.giveColumn()
    random.shuffle(columnNum8)
    for i in xrange(numOfDataPoints):        
        print '\t'.join(map(str, [
            str(1),
        rootColumn[i],
        columnNum2[i], 
        columnNum3[i],
        columnNum4[i],
        columnNum5[i],
        columnNum6[i],
        columnNum7[i],
        columnNum8[i],
        ]))    
    ####
    ####
    ####
    rootColumn = uniformColumn(numOfDataPoints, 4)
    ###
    columnNum2Rule = {
        1:{1:.1, 2:.1, 3:.1, 4:.7},
        2:{1:.1, 2:.1, 3:.7, 4:.1},
        3:{1:.1, 2:.7, 3:.7, 4:.1},
        4:{1:.7, 2:.1, 3:.1, 4:.1},        
        }        
    columnNum2Object = Column(rootColumn, columnNum2Rule)
    columnNum2 = columnNum2Object.giveColumn()
    random.shuffle(columnNum2)
    ###    
    columnNum3Rule = {
        1:{1:.1, 2:.1, 3:.1, 4:.7},
        2:{1:.7, 2:.1, 3:.1, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.7, 4:.1},        
        }        
    columnNum3Object = Column(columnNum2, columnNum3Rule)
    columnNum3 = columnNum3Object.giveColumn()
    # random.shuffle(columnNum3)    
    ###    
    columnNum4Rule = {
        1:{1:.1, 2:.1, 3:.1, 4:.7},
        2:{1:.1, 2:.1, 3:.7, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.7, 2:.1, 3:.1, 4:.1},        
        }        
    columnNum4Object = Column(columnNum3, columnNum4Rule)
    columnNum4 = columnNum4Object.giveColumn()
    ###
    columnNum5Rule = {
        1:{1:.1, 2:.1, 3:.7, 4:.1},
        2:{1:.7, 2:.1, 3:.1, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.1, 4:.7},        
        }        
    columnNum5Object = Column(columnNum4, columnNum5Rule)
    columnNum5 = columnNum5Object.giveColumn()
    random.shuffle(columnNum7)    
    ###
    columnNum6Rule = {
        1:{1:.03, 2:.91, 3:.03, 4:.03},
        2:{1:.91, 2:.03, 3:.03, 4:.03},
        3:{1:.03, 2:.03, 3:.03, 4:.91},
        4:{1:.03, 2:.03, 3:.91, 4:.03},        
        }        
    columnNum6Object = Column(columnNum5, columnNum6Rule)
    columnNum6 = columnNum6Object.giveColumn()
    random.shuffle(columnNum7)    
    ###
    columnNum7Rule = {
        1:{1:.1, 2:.1, 3:.1, 4:.7},
        2:{1:.7, 2:.1, 3:.7, 4:.1},
        3:{1:.1, 2:.7, 3:.1, 4:.1},
        4:{1:.1, 2:.1, 3:.7, 4:.1},        
        }        
    columnNum7Object = Column(columnNum6, columnNum7Rule)
    columnNum7 = columnNum7Object.giveColumn()
    ###
    columnNum8Rule = {
        1:{1:.03, 2:.03, 3:.03, 4:.91},
        2:{1:.03, 2:.91, 3:.03, 4:.03},
        3:{1:.03, 2:.03, 3:.91, 4:.03},
        4:{1:.91, 2:.03, 3:.03, 4:.03},        
        }            
    columnNum8Object = Column(columnNum7, columnNum8Rule)
    columnNum8 = columnNum8Object.giveColumn()
    for i in xrange(numOfDataPoints):        
        print '\t'.join(map(str, [
            str(2),
        rootColumn[i],
        columnNum2[i], 
        columnNum3[i],
        columnNum4[i],
        columnNum5[i],
        columnNum6[i],
        columnNum7[i],
        columnNum8[i],
        ]))    
            

def dependencyData():
    produceDependencyData()


def main():
    # nondependencyData()
    # nondependencyData()
    dependencyData()
    dependencyData()
            

if __name__ == '__main__':
    main()
