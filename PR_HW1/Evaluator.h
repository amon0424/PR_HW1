#ifndef _EVALUATOR_H
#define _EVALUATOR_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "FeatureData.h";
#include "Class.h"
#include "Classifier.h"

class Evaluator 
{
private:
	std::vector<Class> _classes;
	std::vector<FeatureData> _trainingData;
	Classifier* _classifier;

public:
	int NumberOfClasses;
	int NumberOfFeatures;

	void Delete();
	void ReadFile(std::string filename);
	std::vector<FeatureData> ReadTestData(std::string filename);
	int Classify(const Classifier& classifier, const FeatureData& x);
	void Train(Classifier& classifier);
	void InitializeClassifier(Classifier& classifier);
};

#endif
