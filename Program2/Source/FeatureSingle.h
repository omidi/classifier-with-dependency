/*
 * FeatureSingle.h
 *
 *  Created on: Jan 22, 2014
 *      Author: omidi
 */

#ifndef FEATURESINGLE_H_
#define FEATURESINGLE_H_

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <boost/algorithm/string.hpp>

using namespace std;

class FeatureSingle {
	std::vector<float> counts;
public:
	FeatureSingle();
	void initilize_features_counts(int);
	void add_count(int);
	float get_count(int);
	virtual ~FeatureSingle();
};

#endif /* FEATURESINGLE_H_ */
