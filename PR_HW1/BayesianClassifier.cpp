#include "BayesianClassifier.h"
#include <fstream>
#include <iostream>
#include "Utility.h"

BayesianClassifier::~BayesianClassifier()
{
	_classesPdf.clear();
}

void BayesianClassifier::Reset()
{
	_classes.clear();
	_classesPdf.clear();
}

void BayesianClassifier::Print()
{	
	for(int i=0; i < _classes.size(); i++)
	{
		Class& c = _classes[i];
		GaussianPdf& pdf = _classesPdf[i];

		std::cout << "Class " << c.ID << std::endl;
		std::cout << "-------" << std::endl;
		std::cout << "Probability: " << c.Probability << std::endl;
		std::cout << "Mean: " << std::endl;
		Utility::PrintMatrix(pdf.Mean, NumberOfFeatures, 1);
		std::cout << "Covariance: " << std::endl;
		Utility::PrintMatrix(pdf.CovarianceMatrix, NumberOfFeatures, NumberOfFeatures);
		std::cout << std::endl;
	}
}

void BayesianClassifier::SetClasses(const Class* classes, int count)
{
	this->Reset();

	for(int i=0; i<count; i++)
	{
		Class c(classes[i].ID);
		_classes.push_back(c);
		_classesPdf.push_back(GaussianPdf(this->NumberOfFeatures));
	}
}
void BayesianClassifier::Train(const FeatureData* trainingData, int count)
{
	std::vector<const FeatureData*>* newTrainingDataOfClasses = new std::vector<const FeatureData*>[this->NumberOfFeatures]();

	for(int i=0; i<_classes.size(); i++)
	{
		Class& c = _classes[i];
		GaussianPdf& pdf = _classesPdf[i];
		//std::vector<const FeatureData*> oldTrainingData = _trainingDataOfClasses[i];

		cvConvertScale(pdf.Mean, pdf.Mean, c.TrainingData.size());
		cvConvertScale(pdf.CovarianceMatrix, pdf.CovarianceMatrix, c.TrainingData.size());

		//newTrainingDataOfClasses.push_back(std::vector<FeatureData*>());
	}

	for(int i=0; i<count; i++)
	{
		const FeatureData& x = trainingData[i];
		newTrainingDataOfClasses[x.ClassID-1].push_back(&x);
		_classes[x.ClassID-1].TrainingData.push_back(&x);
	}

	int totalTrainingData = 0;
	for(int i=0; i<_classes.size(); i++)
	{
		Class& c = _classes[i];
		GaussianPdf& pdf = _classesPdf[i];
		std::vector<const FeatureData*> newTrainingData = newTrainingDataOfClasses[i];

		// compute mean
		for(int j=0; j<newTrainingData.size(); j++)
		{
			const FeatureData* x = newTrainingData[j];

			cvAdd(pdf.Mean, x->FeatureVector, pdf.Mean);
		}
		cvConvertScale(pdf.Mean, pdf.Mean, 1.0 / c.TrainingData.size());

		// compute covariance matrix
		CvMat* xMinusMean = cvCreateMat(NumberOfFeatures, 1, CV_32FC1);
		CvMat* xMinusMeanT = cvCreateMat(1, NumberOfFeatures, CV_32FC1);
		for(int j=0; j<newTrainingData.size(); j++)
		{
			const FeatureData* x = newTrainingData[j];

			cvSub(x->FeatureVector, pdf.Mean, xMinusMean);
			cvTranspose(xMinusMean, xMinusMeanT);
			cvMatMulAdd(xMinusMean, xMinusMeanT,  pdf.CovarianceMatrix,  pdf.CovarianceMatrix);
		}
		cvConvertScale(pdf.CovarianceMatrix, pdf.CovarianceMatrix, 1.0 / c.TrainingData.size());

		// release matrix
		cvReleaseMat(&xMinusMean);
		cvReleaseMat(&xMinusMeanT);

		totalTrainingData += c.TrainingData.size();
	}


	for(int i=0; i<_classes.size(); i++)
	{
		Class& c = _classes[i];
		c.Probability = c.TrainingData.size() / (float)totalTrainingData;
	}
}

int BayesianClassifier::Classify(const FeatureData& x, double* probability) const
{
	double max = 0;
	int maxClass = NULL;

	for(int i=0;i<_classes.size(); i++)
	{
		double p = _classesPdf[i].GetProbability(x) * _classes[i].Probability;
		if( p > max)
		{
			max = p;
			maxClass = i;
		}
		if(probability != NULL)
			probability[i] = p;
	}

	return maxClass + 1;
}