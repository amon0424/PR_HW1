#ifndef _TRAINING_DATA_H
#define _TRAINING_DATA_H

#include <string>
#include <vector>
#include "Class.h"

class FeatureData;
class TrainingData
{
public:
	std::vector<Class> Classes;
	std::vector<FeatureData> Data;
	int NumberOfClasses;
	int NumberOfFeatures;

	TrainingData() : NumberOfClasses(0), NumberOfFeatures(0){}
	TrainingData(FeatureData** data, int count);
	void ReadFile(std::string filename);
	std::vector<FeatureData>* ReadTestData(std::string filename);
};
#endif