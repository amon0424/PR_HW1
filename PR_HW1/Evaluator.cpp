#include "Evaluator.h"

void Evaluator::Delete()
{
	for(int i=0; i< _trainingData.size(); i++)
	{
		_trainingData[i].Delete();
	}
}

void Evaluator::ReadFile(std::string filename)
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


std::vector<FeatureData> Evaluator::ReadTestData(std::string filename)
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

int Evaluator::Classify(const Classifier& classifier, const FeatureData& x)
{
	return classifier.Classify(x);
}

void Evaluator::Train(Classifier& classifier)
{
	classifier.Train(&_trainingData[0], _trainingData.size());
}

void Evaluator::InitializeClassifier(Classifier& classifier)
{
	classifier.SetClasses(&_classes[0], _classes.size());
}