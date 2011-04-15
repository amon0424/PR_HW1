#ifndef _TRAINING_DATA_H
#define _TRAINING_DATA_H

#include <vector>
#include "Class.h"

class TrainingData : public std::vector<FeatureVector*>
{
public:
	void Delete()
	{
		for(int i=0; i<this->size(); i++)
		{
			cvReleaseMat(&(this[i]));
		}
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

};

#endif