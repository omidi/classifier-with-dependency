/*
 * Class.cpp
 *
 *  Created on: Jan 21, 2014
 *      Author: omidi
 */

#include "Class.h"

Class::Class() {
	// TODO Auto-generated constructor stub

}

Class::Class(int number_of_features, Feature& features)
{
	this->features = features;
	this->number_of_features = number_of_features;
	features_pair.resize(number_of_features, vector<FeaturesPair>(number_of_features));
	feature_single.resize(number_of_features);
	for(int i=0; i<number_of_features; i++){
		feature_single[i].initilize_features_counts(features.get_feature_size(i));
		for(int j=0; j<i; j++){
			features_pair[j][i].initilize_features_counts(features.get_feature_size(j),features.get_feature_size(i));
		}
	}
	sample_size = 0.;
	logR_determinant = 0.;
}


void Class::add_to_feature(vector<string> feature_vector){
	for(int i=0; i < number_of_features; i++){
		feature_single[i].add_count(atoi(feature_vector[i+1].c_str()));
		for(int j=0; j < i; j++){
			features_pair[j][i].add_count(atoi(feature_vector[j+1].c_str())
					, atoi(feature_vector[i+1].c_str()));
		}
	}
	sample_size += 1.;
}


float Class::get_feature_pair(int feature_A, int feature_B, int i, int j){
	return features_pair[feature_A][feature_B].get_count(i, j);
}


float Class::get_feature_single(int feature, int item){
	return feature_single[feature].get_count(item);
}


float Class::calculate_logR(int i, int j) {
	double logR_score = 0.0;
	double features_combinations = features.get_feature_size(i) * features.get_feature_size(j);
	double pseudo_count = 1.0 / features_combinations;
	for(int feature_A=0; feature_A < features.get_feature_size(i); feature_A++){
		logR_score -= lgamma(get_feature_single(i, feature_A) + pseudo_count*features.get_feature_size(i));
		logR_score += lgamma(pseudo_count*features.get_feature_size(i));
	}

	for(int feature_B=0; feature_B < features.get_feature_size(j); feature_B++){
		logR_score -= lgamma(get_feature_single(j, feature_B) + pseudo_count*features.get_feature_size(j));
		logR_score += lgamma(pseudo_count*features.get_feature_size(j));
	}

	for(int feature_A=0; feature_A < features.get_feature_size(i); feature_A++){
		for(int feature_B=0; feature_B < features.get_feature_size(j); feature_B++){
			logR_score += lgamma(get_feature_pair( i, j, feature_A, feature_B) + pseudo_count );
			logR_score -= lgamma(pseudo_count);
		}
	}
	// adding the normalization factors
	logR_score += lgamma(sample_size + 1.0);
//	logR_score -= lgamma(1.0);
	return logR_score;
}


void Class::calculate_logR_matrix(){
	logR.resize(number_of_features, vector<double>(number_of_features, 0.0));
	for (int i=0; i < number_of_features ; i++){
		for (int j=0; j < i ; j++){
//			cout << i << '\t' << j ;
			logR[i][j] = logR[j][i] = calculate_logR(j, i);
//			cout << '\t' << logR[j][i] << '\n';
		}
	}
	alpha_exponent = find_alpha_exponent(logR);
//	logR_determinant = determinant(matrix_minor(laplacian_of(re_scale(logR, alpha_exponent))));
	logR_determinant = calculate_determinant(laplacian_of(re_scale(logR, alpha_exponent)), logR.size());
	if (std::isinf(logR_determinant)){
		logR_determinant = 0.;
	}
//	cout << alpha_exponent << '\t' << logR_determinant << '\n';
}


int determinant_sign(const bnu::permutation_matrix<std::size_t>& pm)
{
    int pm_sign=1;
    std::size_t size = pm.size();
    for (std::size_t i = 0; i < size; ++i)
        if (i != pm(i))
            pm_sign *= -1.0; // swap_rows would swap a pair of rows here, so we change sign
    return pm_sign;
}


double Class::determinant( bnu::matrix<double> m ) {
    bnu::permutation_matrix<std::size_t> pm(m.size1());
    double det = 1.0;
    if( bnu::lu_factorize(m,pm) ) {
        det = 0.0;
    } else {
        for(unsigned int i = 0; i < m.size1(); i++)
            det *= m(i,i); // multiply by elements on diagonal
        det = det * determinant_sign( pm );
    }
    return log(det);  // return determinant in Log space
}


std::vector <std::vector<double> > Class::re_scale(std::vector <std::vector<double> > logR, double alpha_exponent){
/*
 * The re-scaling formula is as following on:
 * 		R -> R^alpha
 * 		where alpha = (K*log10)/(log_10(max) - log_10(min))
 * 		K determines the precision, and should be set based on the numerical capability of the machine, in original paper this value
 * 		set as K=5. In this code K refers as PRECISION (constant) class member.
 * 		the function of this re-scaling method is to shrink the data values in a way the their relative difference will be conserved, however in
 * 		smaller scale.
 * 		For more info refers to "Tractable Bayesian Learning of Tree Augmented Naive Bayes Classifiers", Jesus Cerquides and Ramon de Mantaras, 2003
 * */
	std::vector <std::vector<double> > M;

	M.resize(number_of_features, std::vector<double>(number_of_features));
	for (unsigned int i = 0; i < logR.size(); i++){
		for(unsigned int j = 0; j < i; j++){
			M[j][i] = M[i][j] = exp(alpha_exponent*logR[i][j]);
		}
	}
	return M;
}

inline double Class::calculate_determinant(std::vector <std::vector<double> > M, int n){
	Decomposition D;
	double determinant = 1.0;
	determinant = D.determinant(M, n);
	return determinant;
}

std::vector <std::vector <double> > Class::add_with_identity_matrix(std::vector <std::vector<double> > M){
	for(unsigned int i = 0; i < M.size(); i++){
		M[i][i] += 1.0;
	}
	return M;
}

bnu::matrix<double> Class::matrix_minor(std::vector <std::vector<double> > M){
	bnu::matrix<double> res(M.size()-1, M.size()-1);
	for(unsigned int i=0; i < (M.size()-1); i++){
		for(unsigned int j=0; j<(M.size()-1); j++){
			res(i,j) = M[i][j];
		}
	}
	return res;
}

std::vector <std::vector<double> > Class::laplacian_of(std::vector <std::vector<double> > M){
	std::vector <std::vector<double> > L;
	L.resize(M.size(), std::vector<double>(M.size()));
	for(unsigned short int i = 0; i < M.size(); i++){
		double sum = .0;
		for(unsigned short int j = 0; j< M.size(); j++){
			sum += M[i][j];
			L[i][j] = -M[i][j];
		}
		L[i][i] = sum + 1.0; //	I didn't use "- M[i][i]" because logR[i][i] is 0
							 // for adding the identity matrix to the Laplacian matrix
	}
	return L;
}

double Class::find_alpha_exponent(std::vector <std::vector<double> > logR) {
	double PRECISION = 15.0;
	double max = logR[0][1];
	double min = logR[0][1];
	double alpha = 1.;

	for (unsigned int i = 0; i < logR.size(); i++){
		for(unsigned int j = 0; j < i; j++){
			if (logR[i][j] > max)
				max = logR[i][j];
			if (logR[i][j] < min)
				min = logR[i][j];
		}
	}
	if (max  > PRECISION){
		alpha = PRECISION / max;
	}
	return alpha;
}


double Class::membership_test(std::vector<string> feature_vector){
	double pseudo_count = 0.;
	double score = 0.;
	for(int i=0; i<number_of_features; i++){
		pseudo_count = 1. / features.get_feature_size(i);
		score += log(get_feature_single(i, atoi(feature_vector[i+1].c_str())) + pseudo_count);
	}
	score -= number_of_features*log(sample_size + 1.0);
		if (!std::isinf(logR_determinant)){
		std::vector <std::vector<double> > new_logR = logR_plus_one(feature_vector);
		double new_alpha_exponent = find_alpha_exponent(new_logR);
		double new_determinant = calculate_determinant(laplacian_of(re_scale(new_logR, new_alpha_exponent)), new_logR.size()-1);
//		double new_determinant = determinant(matrix_minor(laplacian_of(re_scale(new_logR, new_alpha_exponent))));
		if (std::isinf(new_determinant)){
			new_determinant = 0.;
		}
//		cout << new_alpha_exponent << '\t' << new_determinant << '\t'<< logR_determinant << '\n';
		score += new_determinant;
		score -= logR_determinant;
	}
	return score;
}

double Class::naive_bayes_membershiptest(vector<string> feature_vector){
	double pseudo_count = 0.;
	double score = 0.;
	for(int i=0; i<number_of_features; i++){
		pseudo_count = 1. / features.get_feature_size(i);
		score += log(get_feature_single(i, atoi(feature_vector[i+1].c_str())) + pseudo_count);
	}
	score -= number_of_features*log(sample_size + 1.0);
	return score;
}

std::vector <std::vector<double> > Class::logR_plus_one(std::vector<string> feature_vector){
	/* instead of counting again number of appearances for each letter and generate new n_i and n_ij matrices,
	 * here, logR of the alignment plus new sequence will be calculated according to the following formula:
	 * logR_ij^s = log(R_ij) + log(n^ij_{si,sj} + LAMBDA) - log(n^i_si + 4*LAMBDA) - log(n^j_sj + 4*LAMBDA) - log(n + 16*LAMBDA) + 2log(n+4*LAMBDA)
	 * where, logR_ij^s is an entry of the new dependency matrix and R_ij is an entry of previously computed dependency matrix R
	 * and n^ij, n^i are frequency matrix that we have them from an object of Alignment class.
	*/

	// the original logR replaced with rescaled_R in following
	std::vector <std::vector<double> > logR_new;
	logR_new.resize(number_of_features, std::vector<double>(number_of_features));	// initialization of logR_new matrix
	double pseudo_count = 0.;

	for(int i = 0; i < number_of_features; i++){
		for(int j = 0; j < i; j++){
			pseudo_count = 1.0 / (features.get_feature_size(i) * features.get_feature_size(j));
			logR_new[i][j]  = logR[i][j];

			logR_new[i][j] += log(sample_size + 1.0);

			logR_new[i][j] -= lgamma(get_feature_pair( j, i,
					(atoi(feature_vector[j+1].c_str()) - 1),
					(atoi(feature_vector[i+1].c_str()) - 1) )  + pseudo_count);
			logR_new[i][j] += lgamma(get_feature_pair( j, i,
					(atoi(feature_vector[j+1].c_str()) - 1),
					(atoi(feature_vector[i+1].c_str()) - 1) )  + pseudo_count + 1.0);

			logR_new[i][j] += lgamma(get_feature_single( j,
					(atoi(feature_vector[j+1].c_str()) - 1) ) + pseudo_count*features.get_feature_size(j));
			logR_new[i][j] -= lgamma(get_feature_single( j,
					(atoi(feature_vector[j+1].c_str()) - 1) ) + pseudo_count*features.get_feature_size(j) + 1.0);

			logR_new[i][j] += lgamma(get_feature_single( i,
					(atoi(feature_vector[i+1].c_str()) - 1) ) + pseudo_count*features.get_feature_size(i));
			logR_new[i][j] -= lgamma(get_feature_single( i,
					(atoi(feature_vector[i+1].c_str()) - 1) ) + pseudo_count*features.get_feature_size(i) + 1.0);
			logR_new[j][i] = logR_new[i][j];

//			cout << j << '\t' << i << '\t'  << logR[i][j] << '\t' << logR_new[i][j] << '\n';
		}
	}

	return logR_new;
}


Class::~Class() {
	// TODO Auto-generated destructor stub
}
