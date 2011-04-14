#include <opencv/cv.h>
#include <vector>

typedef std::vector<float> FeatureVector;

class Class
{
	std::vector<FeatureVector> _tranningData;
	int _numberOfFeatures;
public:
	CvMat* Mean;
	CvMat* CovarianceMatrix;
	

	Class(int numberOfFeatures)
	{
		_numberOfFeatures = numberOfFeatures;
		Mean = cvCreateMat(3,1,CV_32FC1);
		
		CovarianceMatrix = cvCreateMat(3,3,CV_32FC1);
	}
	~Class()
	{
		cvReleaseMat(&Mean);
		cvReleaseMat(&CovarianceMatrix);
	}

	void AddTranningData(FeatureVector& featureVector)
	{
		_tranningData.push_back(featureVector);
	}

	void ComputeParamaters()
	{
		//CvMat mean = cvMat(3,1,CV_32FC1);
		for(int i=0; i<_numberOfFeatures; i++)
			Mean->data.fl[i] = 0;

		for(int i=0; i<_tranningData.size(); i++)
		{
			FeatureVector v = _tranningData[i];

			for(int j=0; j<_numberOfFeatures; j++)
				Mean->data.fl[j] += v[j];
		}

		for(int i=0; i<_numberOfFeatures; i++)
			Mean->data.fl[i] = Mean->data.fl[i] / _tranningData.size();
	}
};