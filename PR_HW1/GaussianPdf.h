#ifndef _GAUSSIAN_PDF_H
#define _GAUSSIAN_PDF_H

#include <opencv/cv.h>
#include <vector>

class FeatureData;

class GaussianPdf 
{
public:
	int Dimension;
	CvMat* Mean;
	CvMat* CovarianceMatrix;

	GaussianPdf(int dimension);
	GaussianPdf(const GaussianPdf& x);
	GaussianPdf& operator=(const GaussianPdf& x);
	~GaussianPdf();

	double GetProbability(const FeatureData& x) const;
};
#endif