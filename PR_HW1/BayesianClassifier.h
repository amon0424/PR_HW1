#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "Class.h"

using std::string;

class BayesianClassifier 
{
	std::vector<FeatureVector*> _tranningData;
	std::vector<Class> _classes;

	

	
public:
	int _numberOfClasses;
	int _numberOfFeatures;
	BayesianClassifier()
	{
		
	}
	~BayesianClassifier()
	{
		
		for(int i=0; i < _numberOfClasses; i++)
		{
			_classes[i].Release();
		}
	}

	void PrintClassesInformation()
	{	
		for(int i=0; i < _numberOfClasses; i++)
		{
			Class c = _classes[i];
			c.Print();
		}
	}

	std::vector<FeatureVector*> ReadTestData(string filename)
	{
		std::vector<FeatureVector*> testData;

		std::ifstream inputFile(filename.c_str());

		int numberOfFeatures;
		inputFile >> numberOfFeatures;
		while(!inputFile.eof())
		{
			try
			{
				FeatureVector* featureVector = cvCreateMat(_numberOfFeatures, 1, CV_32FC1);
				for(int i=0; i<numberOfFeatures; i++)
				{
					float value;
					inputFile >> value;
					featureVector->data.fl[i] = value;
				}
				int classID;
				inputFile >> classID;
				testData.push_back(featureVector);
			}
			catch (std::exception e) 
			{
			}
		}
		return testData;
	}

	void ReadFile(string filename)
	{
		std::ifstream inputFile(filename.c_str());

		inputFile >> _numberOfClasses;
		inputFile >> _numberOfFeatures;

		for(int i=0; i < _numberOfClasses; i++)
		{
			_classes.push_back(Class(i+1, _numberOfFeatures));
		}

		int tmp;
		inputFile >> tmp >> tmp >> tmp;

		while(!inputFile.eof())
		{
			try
			{
				FeatureVector* featureVector = cvCreateMat(_numberOfFeatures, 1, CV_32FC1);
				for(int i=0; i<_numberOfFeatures; i++)
				{
					float value;
					inputFile >> value;
					featureVector->data.fl[i] = value;
				}
				int classID;
				inputFile >> classID;

				_classes[classID-1].AddTranningData(featureVector);
			}
			catch (std::exception e) 
			{
			}
		}

		for(int i=0;i<_numberOfClasses; i++)
		{
			_classes[i].ComputeParamaters();
		}
	}

	Class& Classfy(const FeatureVector& x)
	{
		float max = 0;
		Class* maxClass = NULL;

		for(int i=0;i<_numberOfClasses; i++)
		{
			float p = _classes[i].GetProbability(x);
			if( p > max)
			{
				max = p;
				maxClass = &_classes[i];
			}
		}

		return *maxClass;
	}

	void AddTranningData(FeatureVector& featureVector, int classID)
	{
	}
};
