#ifndef _CLASS_H
#define _CLASS_H

#include <vector>
#include <string>

class FeatureData;

class Class
{

public:
	int ID;
	std::string Label;
	float Probability;
	std::vector<const FeatureData*> TrainingData;

	Class(int id) : ID(id) {}
};

#endif