/*
 * Testing.h
 *
 *  Created on: Feb 13, 2014
 *      Author: omidi
 */

#ifndef TESTING_H_
#define TESTING_H_

#include "TrainingSet.h"
#include<fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <boost/algorithm/string.hpp>

using namespace std;

class Testing {
public:
	Testing(TrainingSet&, string);
	virtual ~Testing();
};

#endif /* TESTING_H_ */
