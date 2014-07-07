/*
 * Feature.cpp
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#include "Feature.h"


Feature::Feature() {
	// TODO Auto-generated constructor stub

}

Feature::Feature(string filename){
	ifstream feature_file(filename.c_str(), ios::in);
	if (!feature_file.is_open()){
		cerr << "The Feature file could not be opened." << endl;
		exit(1);
	} else{
		int number_of_lines = 0;
		int feature_id, feature_size;
		string line;
		while ( getline(feature_file, line) )
			number_of_lines++;
		vector<int> features_vector (number_of_lines);
		feature_file.clear();
		feature_file.seekg(0, feature_file.beg);
		while ( !feature_file.eof() ){
			feature_file >> feature_id >> feature_size;
			features_vector[feature_id - 1] = feature_size;
		}
		features = features_vector;
	}
	feature_file.close();
}


Feature::~Feature() {
	// TODO Auto-generated destructor stub
}

int Feature::get_feature_size(int index){
	return features[index];
}

int Feature::get_number_of_features(){
	return features.size();
}
