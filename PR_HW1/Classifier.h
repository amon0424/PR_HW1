#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H
#include <vector>
#include "FeatureData.h";

class Classifier
{
public:
	virtual void Print() {}
	virtual char* GetName() = 0;
	virtual void SetClasses(const Class* classes, int count) = 0;
	virtual void Reset() = 0;
	virtual void Train(const FeatureData* featureData, int count) = 0;
	virtual int Classify(const FeatureData& x) const { return Classify(x, NULL); }
	virtual int Classify(const FeatureData& x, double* probability) const = 0 ;
};

#endif