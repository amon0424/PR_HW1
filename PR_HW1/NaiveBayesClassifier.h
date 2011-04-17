#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "Class.h"
#include "Classifier.h"
#include "GaussianPdf.h"
#include "Utility.h"

using std::string;

class NaiveBayesClassifier : public Classifier
{
	std::vector<Class> _classes;
	std::vector<std::vector<GaussianPdf>> _classesPdfs;
public:
	int NumberOfFeatures;

	NaiveBayesClassifier(int numberOfFeatures) : NumberOfFeatures(numberOfFeatures)
	{
		
	}

	~NaiveBayesClassifier()
	{
		for(int i=0; i < _classesPdfs.size(); i++)
		{
			for(int j=0; j < _classesPdfs[i].size(); j++)
			{
				_classesPdfs[i][j].Delete();
			}
		}
	}

	void PrintClassesInformation()
	{	
		//for(int i=0; i < _classes.size(); i++)
		//{
		//	Class& c = _classes[i];
		//	GaussianPdf& pdf = _classesPdf[i];

		//	std::cout << "Class " << i+1 << std::endl;
		//	std::cout << "Mean: " << std::endl;
		//	Utility::PrintMatrix(pdf.Mean, NumberOfFeatures, 1);
		//	std::cout << "Covariance: " << std::endl;
		//	Utility::PrintMatrix(pdf.CovarianceMatrix, NumberOfFeatures, NumberOfFeatures);
		//	std::cout << std::endl;
		//}
	}
	void Reset()
	{
		for(int i=0; i<_classesPdfs.size(); i++)
		{
			for(int j=0; j < _classesPdfs[i].size(); j++)
			{
				_classesPdfs[i][j].Delete();
			}
		}

		_classes.clear();
		_classesPdfs.clear();
	}

	void SetClasses(const Class* classes, int count)
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
	void Train(const FeatureData* trainingData, int count)
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
				for(int i=0; i<newTrainingData.size(); i++)
				{
					const FeatureData* x = newTrainingData[i];
					tmpX.data.fl[0] = x->FeatureVector->data.fl[j];

					cvAdd(pdf.Mean, &tmpX, pdf.Mean);

					//c.TrainingData.push_back(x);
				}
				cvConvertScale(pdf.Mean, pdf.Mean, 1.0 / c.TrainingData.size());

				// compute covariance matrix
				CvMat* xMinusMean = cvCreateMat(1, 1, CV_32FC1);
				CvMat* xMinusMeanT = cvCreateMat(1, 1, CV_32FC1);
				
				for(int i=0; i<newTrainingData.size(); i++)
				{
					const FeatureData* x = newTrainingData[i];
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

	int Classify(const FeatureData& x) const
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
};
