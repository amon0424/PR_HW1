#ifndef _CLASS_H
#define _CLASS_H

#include <vector>

class FeatureData;

class Class
{

public:
	int ID;
	float Probability;
	std::vector<const FeatureData*> TrainingData;

	Class(int id) : ID(id) {}
};

#endif