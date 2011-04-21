#ifndef _TRAINING_DATA_H
#define _TRAINING_DATA_H

#include <string>
#include <vector>
#include "Class.h"
#include <ctime>
#include <map>
class FeatureData;
class TrainingData
{
public:
	std::vector<Class> Classes;
	std::vector<FeatureData> Data;
	std::map<std::string,int> MapClassLabelToID;

	int NumberOfClasses;
	int NumberOfFeatures;

	TrainingData() : NumberOfClasses(0), NumberOfFeatures(0) { srand(time(0)); }
	TrainingData(Class* classes, int numberOfClasses, FeatureData** data, int count);
	TrainingData PickRandomData(int n);
	void ReadFile(std::string filename);
	std::vector<FeatureData>* ReadTestData(std::string filename);
};
#endif