#ifndef _NAIVE_BAYES_CLASSIFIER_H
#define _NAIVE_BAYES_CLASSIFIER_H

#include <string>
#include <vector>
#include "Class.h"
#include "Classifier.h"
#include "GaussianPdf.h"


using std::string;

class NaiveBayesClassifier : public Classifier
{
	std::vector<Class> _classes;
	std::vector<std::vector<GaussianPdf>> _classesPdfs;
public:
	int NumberOfFeatures;
	NaiveBayesClassifier(int numberOfFeatures) : NumberOfFeatures(numberOfFeatures){}
	~NaiveBayesClassifier();
	char* GetName() { return "Naive Bayes Classifier"; }
	void Print();
	void Reset();
	void SetClasses(const Class* classes, int count);
	void Train(const FeatureData* trainingData, int count);
	int Classify(const FeatureData& x, double* probability) const;
};

#endif