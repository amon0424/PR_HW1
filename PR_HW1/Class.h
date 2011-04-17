#ifndef _CLASS_H
#define _CLASS_H

#include <vector>
#include "FeatureData.h"

class Class
{

public:
	std::vector<const FeatureData*> TrainingData;
	float Probability;
	int ID;

	Class(int id) : ID(id)
	{
	} 
};

#endif