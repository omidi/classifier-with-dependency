/*
 * TrainingSet.cpp
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#include "TrainingSet.h"

TrainingSet::TrainingSet() {
	// TODO Auto-generated constructor stub

}

TrainingSet::TrainingSet(string filename, int number_of_classes, Feature& features){
	ifstream training_file(filename.c_str(), ios::in);  // a file that contains the training set
	if (!training_file.is_open()){
		cerr << "The Training file could not be opened." << endl;
		exit(1);
	} else{
		classes.resize( number_of_classes, Class(features.get_number_of_features(), features) );
		vector<string> elems;
		string line;
		while ( getline(training_file, line) ){
			boost::split(elems, line, boost::is_any_of("\t "));
			classes[atoi(elems[0].c_str()) - 1].add_to_feature(elems);
		}
		for(int i=0; i<number_of_classes; i++)
			classes[i].calculate_logR_matrix();
	}
}


std::vector<double> TrainingSet::class_membership_scores(std::vector<string> feature_vector){
	std::vector<double> scores;
	scores.resize(classes.size(), 0.0);
	for(unsigned int i=0; i<scores.size(); i++){
		scores[i] = classes[i].membership_test(feature_vector);
	}
	return scores;
}


std::vector<double> TrainingSet::naive_bayes_class_membership_scores(std::vector<string> feature_vector){
	std::vector<double> scores;
	scores.resize(classes.size(), 0.0);
	for(unsigned int i=0; i<scores.size(); i++){
		scores[i] = classes[i].naive_bayes_membershiptest(feature_vector);
	}
	return scores;
}


TrainingSet::~TrainingSet() {
	// TODO Auto-generated destructor stub
}
