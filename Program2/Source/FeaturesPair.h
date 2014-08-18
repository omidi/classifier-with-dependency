/*
 * FeaturesPair.h
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#ifndef FEATURESPAIR_H_
#define FEATURESPAIR_H_

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <boost/algorithm/string.hpp>

using namespace std;

class FeaturesPair {
	vector< vector<float> > counts;
public:
	FeaturesPair();
	FeaturesPair(int, int);
	void initilize_features_counts(int, int);
	void add_count(int, int);
	float get_count(int, int);
	int get_size();
	virtual ~FeaturesPair();
};

#endif /* FEATURESPAIR_H_ */
