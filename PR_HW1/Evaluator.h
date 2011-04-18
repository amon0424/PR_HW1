#ifndef _EVALUATOR_H
#define _EVALUATOR_H

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime> 
#include <opencv/cv.h>

#include "FeatureData.h";
#include "Class.h"
#include "Classifier.h"
#include "Utility.h"
class TrainingData;

class Evaluator 
{
	TrainingData* _trainingData;
public:
	Evaluator(TrainingData& trainingData): _trainingData(&trainingData) {}
	void CrossValidate(Classifier& classifier, int k);
	void ResubstitutionValidate(Classifier& classifier);
};

#endif
