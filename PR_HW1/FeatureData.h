#ifndef _FEATURE_DATA_H
#define _FEATURE_DATA_H
#include <opencv/cv.h>
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

	void Delete()
	{
		cvReleaseMat(&FeatureVector);
	}
};
#endif