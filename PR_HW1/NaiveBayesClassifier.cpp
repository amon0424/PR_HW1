#include "NaiveBayesClassifier.h"
#include <iostream>
#include <iomanip>
#include "Utility.h"

NaiveBayesClassifier::~NaiveBayesClassifier()
{
	for(int i=0; i < _classesPdfs.size(); i++)
	{
		_classesPdfs[i].clear();
	}
	
	_classesPdfs.clear();
}

void NaiveBayesClassifier::Print()
{	
	for(int i=0; i < _classes.size(); i++)
	{
		Class& c = _classes[i];

		std::cout << "Class " << c.ID << std::endl;

		for(int j=0; j < _classesPdfs[i].size(); j++)
		{
			GaussianPdf& pdf = _classesPdfs[i][j];
			std::cout << std::fixed << std::setprecision(2) << "Mean: " << cvmGet(pdf.Mean, 0, 0) << "\tCovariance: " << cvmGet(pdf.CovarianceMatrix, 0, 0) << std::endl;
		}

		std::cout << std::endl;
	}
}

void NaiveBayesClassifier::Reset()
{
	for(int i=0; i < _classesPdfs.size(); i++)
	{
		_classesPdfs[i].clear();
	}

	_classes.clear();
	_classesPdfs.clear();
}

void NaiveBayesClassifier::SetClasses(const Class* classes, int count)
{
	this->Reset();

	for(int i=0; i<count; i++)
	{
		_classes.push_back(classes[i]);

		_classesPdfs.push_back(std::vector<GaussianPdf>());
		for(int j=0; j<NumberOfFeatures; j++)
			_classesPdfs[i].push_back(GaussianPdf(1));
	}
}
void NaiveBayesClassifier::Train(const FeatureData* trainingData, int count)
{
	std::vector<const FeatureData*>* newTrainingDataOfClasses = new std::vector<const FeatureData*>[this->NumberOfFeatures]();

	for(int i=0; i<_classes.size(); i++)
	{
		Class& c = _classes[i];

		for(int j=0; j<NumberOfFeatures; j++)
		{
			GaussianPdf& pdf = _classesPdfs[i][j];
			cvConvertScale(pdf.Mean, pdf.Mean, c.TrainingData.size());
			cvConvertScale(pdf.CovarianceMatrix, pdf.CovarianceMatrix, c.TrainingData.size());
		}
	}

	for(int i=0; i<count; i++)
	{
		const FeatureData& x = trainingData[i];
		newTrainingDataOfClasses[x.ClassID-1].push_back(&x);
		_classes[x.ClassID-1].TrainingData.push_back(&x);
	}

	int totalTrainingData = 0;
	float tmpData = 0;
	CvMat tmpX = cvMat(1, 1, CV_32FC1, &tmpData);

	for(int i=0; i<_classes.size(); i++)
	{
		Class& c = _classes[i];
		std::vector<const FeatureData*> newTrainingData = newTrainingDataOfClasses[i];

		for(int j=0; j<NumberOfFeatures; j++)
		{
			GaussianPdf& pdf = _classesPdfs[i][j];

			// compute mean
			for(int k=0; k<newTrainingData.size(); k++)
			{
				const FeatureData* x = newTrainingData[k];
				tmpX.data.fl[0] = x->FeatureVector->data.fl[j];

				cvAdd(pdf.Mean, &tmpX, pdf.Mean);
			}
			cvConvertScale(pdf.Mean, pdf.Mean, 1.0 / c.TrainingData.size());

			// compute covariance matrix
			CvMat* xMinusMean = cvCreateMat(1, 1, CV_32FC1);
			CvMat* xMinusMeanT = cvCreateMat(1, 1, CV_32FC1);

			for(int k=0; k<newTrainingData.size(); k++)
			{
				const FeatureData* x = newTrainingData[k];
				tmpX.data.fl[0] = x->FeatureVector->data.fl[j];

				cvSub(&tmpX, pdf.Mean, xMinusMean);
				cvTranspose(xMinusMean, xMinusMeanT);
				cvMatMulAdd(xMinusMean, xMinusMeanT,  pdf.CovarianceMatrix,  pdf.CovarianceMatrix);
			}
			cvConvertScale(pdf.CovarianceMatrix, pdf.CovarianceMatrix, 1.0 / c.TrainingData.size());

			// release matrix
			cvReleaseMat(&xMinusMean);
			cvReleaseMat(&xMinusMeanT);
		}
		totalTrainingData += c.TrainingData.size();

	}


	for(int i=0; i<_classes.size(); i++)
	{
		Class& c = _classes[i];
		c.Probability = c.TrainingData.size() / (float)totalTrainingData;
	}
}

int NaiveBayesClassifier::Classify(const FeatureData& x) const
{
	float max = 0;
	int maxClass = NULL;
	FeatureData tmpX(1);

	for(int i=0;i<_classes.size(); i++)
	{
		float p = 1;

		for(int j=0; j<NumberOfFeatures; j++)
		{
			const GaussianPdf& pdf = _classesPdfs[i][j];

			tmpX.FeatureVector->data.fl[0] = x.FeatureVector->data.fl[j];
			p *= pdf.GetProbability(tmpX);
		}

		p *= _classes[i].Probability;

		if( p > max)
		{
			max = p;
			maxClass = i;
		}
	}

	return maxClass + 1;
}