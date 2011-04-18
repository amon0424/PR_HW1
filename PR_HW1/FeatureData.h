#ifndef _FEATURE_DATA_H
#define _FEATURE_DATA_H
#include <opencv/cv.h>
#include <iostream>
class FeatureData
{
public:
	int NumberOfFeatures;
	CvMat* FeatureVector;
	int ClassID;

	FeatureData(int nubmerOfFeatures) : NumberOfFeatures(nubmerOfFeatures)
	{
		FeatureVector = cvCreateMat(nubmerOfFeatures, 1, CV_32FC1);
	}
	
	FeatureData(const FeatureData& x)
	{
		this->FeatureVector = cvCreateMat(x.NumberOfFeatures, 1, CV_32FC1);
		cvCopy(x.FeatureVector, this->FeatureVector);
		ClassID = x.ClassID;
		NumberOfFeatures = x.NumberOfFeatures;
	}

	~FeatureData()
	{
		cvReleaseMat(&FeatureVector);
	}

	FeatureData& operator=(const FeatureData& x)
	{
		cvCopy(x.FeatureVector, this->FeatureVector);
		ClassID = x.ClassID;
		NumberOfFeatures = x.NumberOfFeatures;
		return *this;
	}

	void Print()
	{
		std::cout << "("<< FeatureVector->data.fl[0] << "," << FeatureVector->data.fl[1] << "," << FeatureVector->data.fl[2] << "," << FeatureVector->data.fl[3]  << ")";
	}
};
#endif