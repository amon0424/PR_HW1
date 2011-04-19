#include <fstream>
#include "TrainingData.h"
#include "FeatureData.h"
TrainingData::TrainingData(FeatureData** data, int count)
{
	if(count > 0)
	{
		NumberOfFeatures = (*data[0]).NumberOfFeatures;
		
		int maxClass = 0;
		for(int i=0; i<count; i++)
		{
			FeatureData* x = data[i];
			this->Data.push_back(*x);
			if((*x).ClassID > maxClass)
				maxClass = (*x).ClassID;
		}

		NumberOfClasses = maxClass;

		for(int i=0; i<NumberOfClasses; i++)
			Classes.push_back(Class(i+1));
	}
}
void TrainingData::ReadFile(std::string filename)
{
	std::string line;
	std::ifstream inputFile(filename.c_str());

	inputFile >> this->NumberOfClasses;
	inputFile >> this->NumberOfFeatures;

	for(int i=0; i < NumberOfClasses; i++)
	{
		this->Classes.push_back(Class(i+1));
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

		this->Data.push_back(featureData);
		totalData++;
	}
}


std::vector<FeatureData>* TrainingData::ReadTestData(std::string filename)
{
	std::string line;
	std::vector<FeatureData>* testData = new std::vector<FeatureData>();

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

		testData->push_back(featureData);
	}

	return testData;
}
