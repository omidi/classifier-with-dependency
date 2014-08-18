#! /import/bc2/home/nimwegen/GROUP/local/bin/python

from os import system

system ("g++ -O3 -g3 -Wall -o \"DWT_classification\" -I /import/bc2/soft/app/boost/1.42.0/Linux/include  Classifier_feature_dependency.cpp Class.cpp constants.cpp Decomposition.cpp Feature.cpp FeatureSingle.cpp FeaturesPair.cpp Testing.cpp TrainingSet.cpp")

print "The source code has been compiled and the bindary file is called DWT_classification, on the current directory."


