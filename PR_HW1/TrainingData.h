#ifndef _TRAINING_DATA_H
#define _TRAINING_DATA_H

#include <string>
#include <vector>
#include "Class.h"

class TrainingData
{
public:
	std::vector<Class> Classes;
	std::vector<FeatureData> Data;
	int NumberOfClasses;
	int NumberOfFeatures;

	void ReadFile(std::string filename);
	std::vector<FeatureData>* ReadTestData(std::string filename);
};
#endif