//============================================================================
// Name        : Classifier_feature_dependency.cpp
// Author      : Saeed Omidi
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include "Feature.h"
#include "TrainingSet.h"
#include "Testing.h"

using namespace std;

int main(int argc, char* argv[]) {
	if (argc != 5){
		cerr << "This program takes a file containing the cardinality (length) of features, number of classes, a file containing training set, and the test file" << endl;
		cerr << "e.g. Features N Training Test" << endl;
		exit(1);
	}
	int number_of_classes = atoi(argv[2]);
	Feature features(argv[1]);
	TrainingSet training_set(argv[3], number_of_classes, features);
	Testing (training_set, argv[4]);
}
