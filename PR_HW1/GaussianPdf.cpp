#include "GaussianPdf.h"
#include "FeatureData.h"
#include <opencv/cv.h>
#include <limits>
GaussianPdf::GaussianPdf(int dimension)
{
	this->Dimension = dimension;
	this->Mean = cvCreateMat(dimension, 1, CV_32FC1);
	this->CovarianceMatrix = cvCreateMat(dimension, dimension, CV_32FC1);

	// initialize
	for(int i=0; i<dimension; i++)
	{
		this->Mean->data.fl[i] = 0;
		for(int j=0; j<dimension; j++)
		{
			this->CovarianceMatrix->data.fl[i*dimension+j] = 0;
		}
	}

}
GaussianPdf::GaussianPdf(const GaussianPdf& x)
{
	this->Dimension = x.Dimension;
	this->Mean = cvCreateMat(this->Dimension, 1, CV_32FC1);
	this->CovarianceMatrix = cvCreateMat(this->Dimension, this->Dimension, CV_32FC1);

	cvCopy(x.Mean, this->Mean);
	cvCopy(x.CovarianceMatrix, this->CovarianceMatrix);
}

GaussianPdf& GaussianPdf::operator=(const GaussianPdf& x)
{
	this->Dimension = x.Dimension;
	cvCopy(x.Mean, this->Mean);
	cvCopy(x.CovarianceMatrix, this->CovarianceMatrix);

	return *this;
}

GaussianPdf::~GaussianPdf()
{
	cvReleaseMat(&Mean);
	cvReleaseMat(&CovarianceMatrix);
}

double GaussianPdf::GetProbability(const FeatureData& x) const
{
	float l = this->Dimension;
	float det = cvDet(this->CovarianceMatrix);
	float denom = pow(2 * 3.1415926, l / 2.0) * sqrt(det);

	if(denom == 0)
		denom = 0.00000000001;

	CvMat* xMinusMean = cvCreateMat(this->Dimension, 1, CV_32FC1);
	CvMat* xMinusMeanT = cvCreateMat(1, this->Dimension, CV_32FC1);
	CvMat* invCovariance = cvCreateMat(this->Dimension, this->Dimension, CV_32FC1);
	CvMat* tmp = cvCreateMat(1, 1, CV_32FC1);

	cvSub(x.FeatureVector, this->Mean, xMinusMean);
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

	if(p < DBL_MIN)
		p=0;
	if(p != p)	// check nan
		p=0;
	return p;
}