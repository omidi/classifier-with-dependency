/*
 * Testing.cpp
 *
 *  Created on: Feb 13, 2014
 *      Author: omidi
 */

#include "Testing.h"

Testing::Testing(TrainingSet& trainingSet, string filename) {
	ifstream test_file(filename.c_str(), ios::in);  // a file that contains the test set
	if (!test_file.is_open()){
		cerr << "The Training file could not be opened." << endl;
		exit(1);
	} else{
		vector<string> elems;
		string line;
		double evidence = 0.;
		double naive_bayes_evidence = 0.;
		while ( getline(test_file, line) ){
			boost::split(elems, line, boost::is_any_of("\t "));
			std::vector<double> scores = trainingSet.class_membership_scores(elems);
			std::vector<double> naive_bayes_scores = trainingSet.naive_bayes_class_membership_scores(elems);
			naive_bayes_evidence = 0.;
			evidence = 0.;
			for(unsigned int i=0; i<scores.size(); i++){
				evidence += exp(scores[i]);
				naive_bayes_evidence += exp(naive_bayes_scores[i]);
			}
			for(unsigned int i=0; i<scores.size(); i++){
				cout <<	elems[0] << '\t' <<
						i + 1 << '\t' <<
						exp( scores[i] - log(evidence) ) << '\t' <<
						exp( naive_bayes_scores[i] - log(naive_bayes_evidence) ) <<
						'\n';
			}
//			cout << '\n';
		}
	}
}

Testing::~Testing() {
	// TODO Auto-generated destructor stub
}
