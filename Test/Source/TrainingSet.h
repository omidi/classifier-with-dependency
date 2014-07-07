/*
 * TrainingSet.h
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#ifndef TRAININGSET_H_
#define TRAININGSET_H_

#include<fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "Feature.h"
#include "Class.h"

using namespace std;


class TrainingSet {
	vector<Class> classes;
public:
	TrainingSet();
	TrainingSet(string, int, Feature&);
	std::vector<double> class_membership_scores(std::vector<string>);
	std::vector<double> naive_bayes_class_membership_scores(std::vector<string>);
	virtual ~TrainingSet();
};

#endif /* TRAININGSET_H_ */
