#ifndef _BAYESIAN_CLASSIFIER_H
#define _BAYESIAN_CLASSIFIER_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "Class.h"
#include "Classifier.h"
#include "GaussianPdf.h"
#include "Utility.h"

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
	void PrintClassesInformation()
	{	
		for(int i=0; i < _classes.size(); i++)
		{
			Class& c = _classes[i];
			GaussianPdf& pdf = _classesPdf[i];

			std::cout << "Class " << i+1 << std::endl;
			std::cout << "Mean: " << std::endl;
			Utility::PrintMatrix(pdf.Mean, NumberOfFeatures, 1);
			std::cout << "Covariance: " << std::endl;
			Utility::PrintMatrix(pdf.CovarianceMatrix, NumberOfFeatures, NumberOfFeatures);
			std::cout << std::endl;
		}
	}
	void Reset();
	void SetClasses(const Class* classes, int count);
	void Train(const FeatureData* trainingData, int count);
	int Classify(const FeatureData& x) const;
};

#endif