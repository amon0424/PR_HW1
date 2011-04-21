#include <fstream>
#include "TrainingData.h"
#include "FeatureData.h"
#include <ctime>
#include <vector>
TrainingData::TrainingData(Class* classes, int numberOfClasses, FeatureData** data, int count)
{
	srand(time(0));
	if(count > 0)
	{
		NumberOfFeatures = (*data[0]).NumberOfFeatures;
		NumberOfClasses = numberOfClasses;

		for(int i=0; i<NumberOfClasses; i++)
		{
			Class c = Class(i+1);
			c.Label = classes[i].Label;
			Classes.push_back(c);
		}

		for(int i=0; i<count; i++)
		{
			FeatureData* x = data[i];
			this->Data.push_back(*x);
			this->Classes[x->ClassID-1].TrainingData.push_back(x);
		}

		for(int i=0; i<NumberOfClasses; i++)
		{
			this->Classes[i].Probability = this->Classes[i].TrainingData.size() / (float)count;
		}
	}
}
TrainingData TrainingData::PickRandomData(int n)
{
	if(n > this->Data.size())
	{
		std::cout << "Error: N is larger than the size of training data." << std::endl;
		return TrainingData();
	}

	std::vector<FeatureData*> tmpTrainingData;
	std::vector<FeatureData*>* trainingDataOfClasses = new std::vector<FeatureData*>[this->Classes.size()];
	

	for(int i=0; i<this->Data.size() ; i++)
	{
		tmpTrainingData.push_back(&this->Data[i]);
		trainingDataOfClasses[this->Data[i].ClassID-1].push_back(&this->Data[i]);
	}

	// decide the pick number of each classes
	int* pickNumberOfClasses = new int[this->Classes.size()];
	int remain = n;

	for(int i=0;i<this->Classes.size(); i++)
	{
		pickNumberOfClasses[i] = n * this->Classes[i].Probability;
		remain -= pickNumberOfClasses[i];
	}

	int tmp = 0;
	while(remain > 0)
	{
		pickNumberOfClasses[tmp]++;
		remain--;
		tmp = (tmp+1) % this->Classes.size();
	}

	std::vector<FeatureData*> trainingData;
	

	// shuffle training data
	for(int c=0; c<this->Classes.size(); c++)
	{
		for(int i=0; i<pickNumberOfClasses[c] ; i++)
		{
			int r = i + (rand() % (trainingDataOfClasses[c].size()-i));

			trainingData.push_back(trainingDataOfClasses[c][r]);
			trainingDataOfClasses[c][r] = trainingDataOfClasses[c][i];
		}
	}

	return TrainingData(&this->Classes[0], this->Classes.size(), &trainingData[0], n);
}
void TrainingData::ReadFile(std::string filename)
{
	std::string line;
	std::ifstream inputFile(filename.c_str());

	std::getline(inputFile, line);
	std::stringstream lineStream(line);
	lineStream >> this->NumberOfClasses;
	lineStream >> this->NumberOfFeatures;

	for(int i=0; i < NumberOfClasses; i++)
	{
		this->Classes.push_back(Class(i+1));
		std::getline(inputFile, line);
	}

	int totalData = 0;
	bool readEnd = false;
	while(!inputFile.eof())
	{
		std::getline(inputFile, line);
		std::stringstream lineStream = std::stringstream(line);

		FeatureData featureData(this->NumberOfFeatures);

		for(int i=0; i<this->NumberOfFeatures; i++)
		{
			if(lineStream.eof())
			{
				readEnd = true;
				break;
			}

			float value;
			lineStream >> value;
			featureData.FeatureVector->data.fl[i] = value;
		}
		if(readEnd)
			break;

		std::string classLabel;
		lineStream >> classLabel;

		//find class id
		std::map<std::string,int>::iterator it;
		if((it=this->MapClassLabelToID.find(classLabel)) != this->MapClassLabelToID.end())
			featureData.ClassID = (*it).second;
		else
		{
			featureData.ClassID = this->MapClassLabelToID.size() + 1;
			this->MapClassLabelToID[classLabel] = featureData.ClassID;
			this->Classes[featureData.ClassID-1].Label = classLabel;
		}

		this->Data.push_back(featureData);
		this->Classes[featureData.ClassID-1].TrainingData.push_back(&this->Data.back());

		totalData++;
	}

	for(int i=0; i < NumberOfClasses; i++)
	{
		this->Classes[i].Probability = this->Classes[i].TrainingData.size() / (float)totalData;
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
			if(lineStream.eof())
			{
				readEnd = true;
				break;
			}

			lineStream >> featureData.FeatureVector->data.fl[i];
		}
		if(readEnd)
			break;

		lineStream >> featureData.ClassID;

		testData->push_back(featureData);
	}

	return testData;
}
