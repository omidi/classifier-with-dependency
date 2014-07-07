/*
 * FeaturesPair.cpp
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#include "FeaturesPair.h"

FeaturesPair::FeaturesPair() {
	// TODO Auto-generated constructor stub
}

FeaturesPair::FeaturesPair(int feature_A, int feature_B){
	counts.resize(feature_A, vector <float>(feature_B) );
}

void FeaturesPair::initilize_features_counts(int feature_A, int feature_B){
	counts.resize(feature_A, vector <float>(feature_B) );
}

void FeaturesPair::add_count(int feature_A, int feature_B){
	counts[feature_A - 1][feature_B - 1] += 1.0;
}

float FeaturesPair::get_count(int i, int j){
	return counts[i][j];
}

int FeaturesPair::get_size(){
	return counts.size();
}


FeaturesPair::~FeaturesPair() {
	// TODO Auto-generated destructor stub
}
