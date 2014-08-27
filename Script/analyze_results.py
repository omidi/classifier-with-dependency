
def specificity_sensitivity(results):
    dep_FP, dep_TP = 0, 0
    dep_FN, dep_TN = 0, 0
    naive_FP, naive_TP = 0.0001, 0.0001
    naive_FN, naive_TN = 0, 0
    dep_misclassification_probability, naive_misclassification_probability= 0., 0.
    dependency_decision = lambda v,w: v if v[2]>w[2] else w
    naive_bayes_decision = lambda v,w: v if v[3]>w[3] else w
    for i in xrange(0, len(results), 2):
        dependency =  dependency_decision(results[i], results[i+1])
        naive_bayes = naive_bayes_decision(results[i], results[i+1])
        if dependency[0] == dependency[1]:
            if dependency[0] == 1:
                dep_TP += 1
            else:
                dep_TN += 1
            dep_misclassification_probability += (1.0 - dependency[2])
        else:
            dep_misclassification_probability += dependency[2]
            if dependency[0] == 1:
                dep_FN += 1
            else:
                dep_FP += 1
        if naive_bayes[0] == naive_bayes[1]:            
            if naive_bayes[0] == 1:
                naive_TP += 1
            else:
                naive_TN += 1
            naive_misclassification_probability += (1.0 - naive_bayes[3])
        else:
            naive_misclassification_probability += naive_bayes[3]
            if naive_bayes[0] == 1:
                naive_FN += 1
            else:
                naive_FP += 1
    try:
        dep_sensitivity = float(dep_TP) / (dep_TP + dep_FN)
    except ZeroDivisionError:
        dep_sensitivity = 0
    try:        
        dep_specificity = float(dep_TN) / (dep_FP + dep_TN)
    except ZeroDivisionError:
        dep_specificity = 0
    try:
        dep_precision = float(dep_TP) / (dep_TP + dep_FP)
    except ZeroDivisionError:
        dep_precision = 0
    try:
        dep_NPV = float(dep_TN) / (dep_TN + dep_FN)
    except ZeroDivisionError:
        dep_NPV = 0        
    naive_sensitivity = float(naive_TP) / (naive_TP + naive_FN)
    naive_specificity = float(naive_TN) / (naive_FP + naive_TN)
    naive_precision = float(naive_TP) / (naive_TP + naive_FP)
    naive_NPV = float(naive_TN) / (naive_TN + naive_FN)

    if dep_sensitivity==1. or dep_specificity==1.:
        # print results
        None

    print 'Sensitivity: ', dep_sensitivity, '\t', naive_sensitivity
    print 'Specificity: ', dep_specificity, '\t', naive_specificity
    print 'Precision: ', dep_precision, '\t', naive_precision
    print 'Negative predictive value: ', dep_NPV, '\t', naive_NPV
    print 'Misclassification probability', dep_misclassification_probability / (len(results) / 2), '\t', \
          naive_misclassification_probability / (len(results) / 2)
    print
    return True


def loss_function(results):
    dep_score = 0.
    naive_score = 0.
    dependency_decision = lambda v,w: v if v[2]>w[2] else w
    naive_bayes_decision = lambda v,w: v if v[3]>w[3] else w
    for i in xrange(0, len(results), 2):        
        dependency =  dependency_decision(results[i], results[i+1])
        naive_bayes = naive_bayes_decision(results[i], results[i+1])
        
        if dependency[0] == dependency[1]:
            dep_score += dependency[2]
        else:
            dep_score -= dependency[2]
        if naive_bayes[0] == naive_bayes[1]:
            naive_score += naive_bayes[2]
        else:
            naive_score -= naive_bayes[2]

    print dep_score, '\t', naive_score
    return True
