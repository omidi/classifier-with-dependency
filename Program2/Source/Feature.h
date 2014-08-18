/*
 * Feature.h
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#ifndef FEATURE_H_
#define FEATURE_H_

#include <iostream>
#include <stdlib.h>
#include<fstream>
#include <vector>

using namespace std;

class Feature {
	std::vector <int> features;
public:
	Feature();
	Feature(string);
	virtual ~Feature();
	int get_feature_size(int);
	int get_number_of_features();
};

#endif /* FEATURE_H_ */
