#ifndef _CLASS_H
#define _CLASS_H

#include <opencv/cv.h>
#include <vector>
#include <iostream>

typedef CvMat FeatureVector;

class Class
{
	std::vector<FeatureVector*> _tranningData;
	int _numberOfFeatures;

	void PrintMatrix(CvMat* matrix, int rows, int cols)
	{
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
				printf("%.1f ", matrix->data.fl[i*cols+j]);
			}
			printf("\n");
		}
	}
public:
	CvMat* Mean;
	CvMat* CovarianceMatrix;
	float Probability;
	int ID;

	Class(int id, int numberOfFeatures)
	{
		this->ID = id;
		_numberOfFeatures = numberOfFeatures;
		Mean = cvCreateMat(numberOfFeatures, 1, CV_32FC1);
		CovarianceMatrix = cvCreateMat(numberOfFeatures, numberOfFeatures, CV_32FC1);
	} 
	~Class()
	{
	}

	void Print()
	{
		std::cout << "Class " << this->ID << std::endl;
		std::cout << "Mean: " << std::endl;
		PrintMatrix(this->Mean, _numberOfFeatures, 1);
		std::cout << "Covariance: " << std::endl;
		PrintMatrix(this->CovarianceMatrix, _numberOfFeatures, _numberOfFeatures);
		std::cout << std::endl;
	}

	void AddTrainingData(FeatureVector* featureVector)
	{
		_tranningData.push_back(featureVector);
	}

	void ComputeParamaters()
	{
		CvMat mean = cvMat(3,1,CV_32FC1);
		for(int i=0; i<_numberOfFeatures; i++)
		{
			this->Mean->data.fl[i] = 0;
			for(int j=0; j<_numberOfFeatures; j++)
			{
				this->CovarianceMatrix->data.fl[i*_numberOfFeatures+j] = 0;
			}
		}

		for(int i=0; i<_tranningData.size(); i++)
		{
			FeatureVector* x = _tranningData[i];

			cvAdd(this->Mean, x, this->Mean);
		}
		cvConvertScale(this->Mean, this->Mean, 1.0 / _tranningData.size());

		// compute covariance matrix
		CvMat* xMinusMean = cvCreateMat(_numberOfFeatures, 1, CV_32FC1);
		CvMat* xMinusMeanT = cvCreateMat(1, _numberOfFeatures, CV_32FC1);
		
		for(int i=0; i<_tranningData.size(); i++)
		{
			FeatureVector* x = _tranningData[i];

			cvSub(x, this->Mean, xMinusMean);
			cvTranspose(xMinusMean, xMinusMeanT);
			cvMatMulAdd(xMinusMean, xMinusMeanT,  this->CovarianceMatrix,  this->CovarianceMatrix);
		}
		cvConvertScale(this->CovarianceMatrix, this->CovarianceMatrix, 1.0 / _tranningData.size());
		cvReleaseMat(&xMinusMean);
		cvReleaseMat(&xMinusMeanT);

	}

	float GetProbability(const FeatureVector& x)
	{
		float l = _numberOfFeatures;
		float det = cvDet(this->CovarianceMatrix);
		float denom = pow(2 * 3.1415926, l / 2.0) * sqrt(det);

		if(denom == 0)
			denom = 0.00000000001;

		CvMat* xMinusMean = cvCreateMat(_numberOfFeatures, 1, CV_32FC1);
		CvMat* xMinusMeanT = cvCreateMat(1, _numberOfFeatures, CV_32FC1);
		CvMat* invCovariance = cvCreateMat(_numberOfFeatures, _numberOfFeatures, CV_32FC1);
		CvMat* tmp = cvCreateMat(1, 1, CV_32FC1);
		
		cvSub(&x, this->Mean, xMinusMean);
		cvTranspose(xMinusMean, xMinusMeanT);
		cvInvert(this->CovarianceMatrix, invCovariance);

		//(x-u)^T*invCovar
		cvmMul(xMinusMeanT, invCovariance, xMinusMeanT);
		//(x-u)^T*invCovar * (x-u)
		cvmMul(xMinusMeanT, xMinusMean, tmp);

		float p = exp( -0.5 * tmp->data.fl[0]) / denom;

		cvReleaseMat(&xMinusMean);
		cvReleaseMat(&xMinusMeanT);
		cvReleaseMat(&invCovariance);
		cvReleaseMat(&tmp);

		return p;
	}

	void Release()
	{
		cvReleaseMat(&Mean);
		cvReleaseMat(&CovarianceMatrix);

		for(int i=0; i<_tranningData.size(); i++)
		{
			cvReleaseMat(&_tranningData[i]);
		}
	}
};

#endif