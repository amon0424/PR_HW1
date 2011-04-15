#ifndef _GAUSSIAN_PDF_H
#define _GAUSSIAN_PDF_H

#include "TrainingData.h"
#include <opencv/cv.h>
#include <vector>

class GaussianPdf 
{
public:
	int Dimension;
	CvMat* Mean;
	CvMat* CovarianceMatrix;
	TrainingData TraningData;

	GaussianPdf(int dimension)
	{
		this->Dimension = dimension;
		this->Mean = cvCreateMat(numberOfFeatures, 1, CV_32FC1);
		this->CovarianceMatrix = cvCreateMat(numberOfFeatures, numberOfFeatures, CV_32FC1);
	}

	void Delete()
	{
		cvReleaseMat(&Mean);
		cvReleaseMat(&CovarianceMatrix);
	}

	void SetTrainingData(const TrainingData& traningData)
	{
		this->TraningData.clear();
		for(int i=0; i<traningData.size(); i++)
			this->TraningData.push_back(traningData[i]);
	}

	void ComputeParamaters()
	{
		// initialize
		for(int i=0; i<dimension; i++)
		{
			this->Mean->data.fl[i] = 0;
			for(int j=0; j<dimension; j++)
			{
				this->CovarianceMatrix->data.fl[i*dimension+j] = 0;
			}
		}

		// compute mean
		for(int i=0; i<_tranningData.size(); i++)
		{
			FeatureVector* x = _tranningData[i];

			cvAdd(this->Mean, x, this->Mean);
		}
		cvConvertScale(this->Mean, this->Mean, 1.0 / _tranningData.size());

		// compute covariance matrix
		CvMat* xMinusMean = cvCreateMat(dimension, 1, CV_32FC1);
		CvMat* xMinusMeanT = cvCreateMat(1, dimension, CV_32FC1);
		
		for(int i=0; i<_tranningData.size(); i++)
		{
			FeatureVector* x = _tranningData[i];

			cvSub(x, this->Mean, xMinusMean);
			cvTranspose(xMinusMean, xMinusMeanT);
			cvMatMulAdd(xMinusMean, xMinusMeanT,  this->CovarianceMatrix,  this->CovarianceMatrix);
		}
		cvConvertScale(this->CovarianceMatrix, this->CovarianceMatrix, 1.0 / _tranningData.size());

		// release matrix
		cvReleaseMat(&xMinusMean);
		cvReleaseMat(&xMinusMeanT);
	}
};
#endif