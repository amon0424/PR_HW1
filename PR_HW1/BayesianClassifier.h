#ifndef _BAYESIAN_CLASSIFIER_H
#define _BAYESIAN_CLASSIFIER_H

#include <string>
#include <vector>
#include "Class.h"
#include "Classifier.h"
#include "GaussianPdf.h"

using std::string;

class BayesianClassifier : public Classifier
{
	std::vector<Class> _classes;
	std::vector<GaussianPdf> _classesPdf;
public:
	int NumberOfFeatures;

	BayesianClassifier(int numberOfFeatures) : NumberOfFeatures(numberOfFeatures) {}

	~BayesianClassifier();
	char* GetName() { return "Bayesian Classifier"; }
	void Print();
	void Reset();
	void SetClasses(const Class* classes, int count);
	void Train(const FeatureData* trainingData, int count);
	int Classify(const FeatureData& x, double* probability) const;
};

#endif