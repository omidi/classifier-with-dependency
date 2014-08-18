/*
 * FeatureSingle.cpp
 *
 *  Created on: Jan 22, 2014
 *      Author: omidi
 */

#include "FeatureSingle.h"

FeatureSingle::FeatureSingle() {
	// TODO Auto-generated constructor stub

}

void FeatureSingle::initilize_features_counts(int feature_size){
	counts.resize(feature_size, 0.0);
}

void FeatureSingle::add_count(int item){
	counts[item-1] += 1.0;
}

float FeatureSingle::get_count(int item){
	return counts[item];
}


FeatureSingle::~FeatureSingle() {
	// TODO Auto-generated destructor stub
}
