#include "BayesianClassifier.h"

BayesianClassifier::~BayesianClassifier()
{
	for(int i=0; i < _classesPdf.size(); i++)
	{
		_classesPdf[i].Delete();
	}
}

void BayesianClassifier::Reset()
{
	for(int i=0; i<_classes.size(); i++)
	{
		_classesPdf[i].Delete();
	}

	_classes.clear();
	_classesPdf.clear();
}

void BayesianClassifier::SetClasses(const Class* classes, int count)
{
	this->Reset();

	for(int i=0; i<count; i++)
	{
		_classes.push_back(classes[i]);
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
		for(int i=0; i<newTrainingData.size(); i++)
		{
			const FeatureData* x = newTrainingData[i];

			cvAdd(pdf.Mean, x->FeatureVector, pdf.Mean);

			//c.TrainingData.push_back(x);
		}
		cvConvertScale(pdf.Mean, pdf.Mean, 1.0 / c.TrainingData.size());

		// compute covariance matrix
		CvMat* xMinusMean = cvCreateMat(NumberOfFeatures, 1, CV_32FC1);
		CvMat* xMinusMeanT = cvCreateMat(1, NumberOfFeatures, CV_32FC1);

		for(int i=0; i<newTrainingData.size(); i++)
		{
			const FeatureData* x = newTrainingData[i];

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

int BayesianClassifier::Classify(const FeatureData& x) const
{
	float max = 0;
	int maxClass = NULL;

	for(int i=0;i<_classes.size(); i++)
	{
		float p = _classesPdf[i].GetProbability(x) * _classes[i].Probability;
		if( p > max)
		{
			max = p;
			maxClass = i;
		}
	}

	return maxClass + 1;
}