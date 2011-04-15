#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "Class.h"
#include "TrainingData.h"

using std::string;

class BayesianClassifier 
{
	std::vector<FeatureVector*> _tranningData;
	std::vector<Class> _classes;

public:
	int NumberOfClasses;
	int NumberOfFeatures;

	BayesianClassifier()
	{
		
	}

	~BayesianClassifier()
	{
		for(int i=0; i < NumberOfClasses; i++)
		{
			_classes[i].Release();
		}
	}

	void PrintClassesInformation()
	{	
		for(int i=0; i < NumberOfClasses; i++)
		{
			Class c = _classes[i];
			c.Print();
		}
	}

	std::vector<FeatureVector*> ReadTestData(string filename)
	{
		std::string line;
		std::vector<FeatureVector*> testData;

		std::ifstream inputFile(filename.c_str());

		int numberOfFeatures;
		inputFile >> numberOfFeatures;
		std::getline(inputFile, line);

		bool readEnd = false;
		while(!inputFile.eof())
		{
			std::getline(inputFile, line);
 			std::istringstream lineStream(line);

			FeatureVector* featureVector = cvCreateMat(NumberOfFeatures, 1, CV_32FC1);
			for(int i=0; i<NumberOfFeatures; i++)
			{
				float value;
				lineStream >> value;
				if(lineStream.eof())
				{
					readEnd = true;
					break;
				}
				featureVector->data.fl[i] = value;
			}
			if(readEnd)
				break;

			int classID;
			lineStream >> classID;

			testData.push_back(featureVector);
		}
		return testData;
	}


	void ReadFile(string filename)
	{
		std::string line;
		std::ifstream inputFile(filename.c_str());

		inputFile >> NumberOfClasses;
		inputFile >> NumberOfFeatures;

		for(int i=0; i < NumberOfClasses; i++)
		{
			_classes.push_back(Class(i+1, NumberOfFeatures));
		}

		int tmp;
		inputFile >> tmp >> tmp >> tmp;
		std::getline(inputFile, line);

		int totalData = 0;
		bool readEnd = false;
		while(!inputFile.eof())
		{
			std::getline(inputFile, line);
 			std::istringstream lineStream(line);

			FeatureVector* featureVector = cvCreateMat(NumberOfFeatures, 1, CV_32FC1);
			for(int i=0; i<NumberOfFeatures; i++)
			{
				float value;
				lineStream >> value;
				if(lineStream.eof())
				{
					readEnd = true;
					break;
				}
				featureVector->data.fl[i] = value;
			}
			if(readEnd)
				break;

			int classID;
			lineStream >> classID;

			_classes[classID-1].AddTrainingData(featureVector);
			totalData++;
		}

		for(int i=0;i<NumberOfClasses; i++)
		{
			_classes[i].ComputeParamaters();
		}
	}

	void Train(const TrainingData& traningData)
	{

	}

	Class& Classfy(const FeatureVector& x)
	{
		float max = 0;
		Class* maxClass = NULL;

		for(int i=0;i<NumberOfClasses; i++)
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
};
