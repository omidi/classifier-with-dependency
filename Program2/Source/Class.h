/*
 * Class.h
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#ifndef CLASS_H_
#define CLASS_H_

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "FeaturesPair.h"
#include "FeatureSingle.h"
#include "Feature.h"
#include "Decomposition.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

using namespace std;
namespace bnu = boost::numeric::ublas;

class Class {
	std::vector <std::vector<FeaturesPair> > features_pair;
	std::vector<FeatureSingle> feature_single;
	std::vector <std::vector<double> > logR;
	float sample_size;
	int number_of_features;
	Feature features;
	double logR_determinant;
	double alpha_exponent;
	float calculate_logR(int, int);
	double determinant( bnu::matrix<double> );
	std::vector <std::vector<double> > laplacian_of(std::vector <std::vector<double> >);
	bnu::matrix<double> matrix_minor(std::vector <std::vector<double> >);
	std::vector <std::vector<double> > re_scale(std::vector <std::vector<double> >, double);
	double find_alpha_exponent(std::vector <std::vector<double> >);
	std::vector <std::vector<double> > logR_plus_one(std::vector<string>);

public:
	Class();
	Class(int, Feature&);
	void add_to_feature(vector<string>);
	float get_feature_pair(int, int, int, int);
	float get_feature_single(int, int);
	void calculate_logR_matrix();
	inline double calculate_determinant(std::vector <std::vector<double> >, int);
	double membership_test(vector<string>);
	double naive_bayes_membershiptest(vector<string>);
	std::vector <std::vector<double> > add_with_identity_matrix(std::vector <std::vector<double> >);
	virtual ~Class();
};

#endif /* CLASS_H_ */
