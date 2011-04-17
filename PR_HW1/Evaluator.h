#ifndef _EVALUATOR_H
#define _EVALUATOR_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "FeatureData.h";
#include "Class.h"
#include "Classifier.h"

class Evaluator 
{
private:
	std::vector<Class> _classes;
	std::vector<FeatureData> _trainingData;
	Classifier* _classifier;

public:
	int NumberOfClasses;
	int NumberOfFeatures;

	void Delete()
	{
		for(int i=0; i< _trainingData.size(); i++)
		{
			_trainingData[i].Delete();
		}
	}

	void ReadFile(string filename)
	{
		std::string line;
		std::ifstream inputFile(filename.c_str());

		inputFile >> this->NumberOfClasses;
		inputFile >> this->NumberOfFeatures;

		for(int i=0; i < NumberOfClasses; i++)
		{
			_classes.push_back(Class(i+1));
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

			FeatureData featureData(this->NumberOfFeatures);

			for(int i=0; i<this->NumberOfFeatures; i++)
			{
				lineStream >> featureData.FeatureVector->data.fl[i];
				if(lineStream.eof())
				{
					readEnd = true;
					break;
				}
			}
			if(readEnd)
				break;

			lineStream >> featureData.ClassID;

			_trainingData.push_back(featureData);
			totalData++;
		}
	}

	
	std::vector<FeatureData> ReadTestData(string filename)
	{
		std::string line;
		std::vector<FeatureData> testData;

		std::ifstream inputFile(filename.c_str());

		int numberOfFeatures;
		inputFile >> numberOfFeatures;
		std::getline(inputFile, line);

		bool readEnd = false;
		while(!inputFile.eof())
		{
			std::getline(inputFile, line);
 			std::istringstream lineStream(line);

			FeatureData featureData(NumberOfFeatures);
			for(int i=0; i<this->NumberOfFeatures; i++)
			{
				lineStream >> featureData.FeatureVector->data.fl[i];
				if(lineStream.eof())
				{
					readEnd = true;
					break;
				}
			}
			if(readEnd)
				break;

			lineStream >> featureData.ClassID;

			testData.push_back(featureData);
		}
		return testData;
	}

	int Classify(const Classifier& classifier, const FeatureData& x)
	{
		return classifier.Classify(x);
	}

	void Train(Classifier& classifier)
	{
		classifier.Train(&_trainingData[0], _trainingData.size());
	}

	void InitializeClassifier(Classifier& classifier)
	{
		classifier.SetClasses(&_classes[0], _classes.size());
	}
};

#endif
