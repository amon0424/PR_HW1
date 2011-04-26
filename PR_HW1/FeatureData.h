#ifndef _FEATURE_DATA_H
#define _FEATURE_DATA_H
#include <opencv/cv.h>
#include <iostream>
#include <iomanip>
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
		std::stringstream s;
		//std::cout.width(20);
		//s << "(" << FeatureVector->data.fl[0] << "," << FeatureVector->data.fl[1] << "," << FeatureVector->data.fl[2] << "," << FeatureVector->data.fl[3]  << ")";
		
		s << "(";
		for(int i=0; i<NumberOfFeatures; i++)
		{
			if((float)((int)FeatureVector->data.fl[i]) == FeatureVector->data.fl[i])
				s << std::left << std::setw(4) << FeatureVector->data.fl[i];
			else
				s << std::left << std::setprecision(3) << std::setw(4) << FeatureVector->data.fl[i];

			if(i!=NumberOfFeatures-1)
				s << ",";
		}
		s << ")";
		std::cout << std::left << std::setw(22) << s.str();
		//std::cout << std::left /*<< std::setw(20)*/ << s.str();
	}
};
#endif