#include <string>
#include <iostream>
#include <vector>
#include <opencv/cv.h>

#include "BayesianClassifier.h"
#include "NaiveBayesClassifier.h"
#include "Evaluator.h"
#include "TrainingData.h"
#include "Utility.h"

int main(int argc, char* argv[])
{
	TrainingData trainingData;
	trainingData.ReadFile(std::string("data-iris.txt"));
	
	// Classifiers
	BayesianClassifier bc(trainingData.NumberOfFeatures);
	NaiveBayesClassifier nc(trainingData.NumberOfFeatures);

	Classifier* classifiers[2] = { &bc, &nc };

	// Begin classify
	std::vector<FeatureData>& testData = *trainingData.ReadTestData("testdata.txt");
	for(int classifierID = 0; classifierID < 2; classifierID++)
	{
		Classifier& classifier = *classifiers[classifierID];
		classifier.SetClasses(&trainingData.Classes[0], trainingData.Classes.size());
		classifier.Train(&trainingData.Data[0], trainingData.Data.size());
		classifier.Print();

		std::cout << "Using " << classifier.GetName() << std::endl;

		for(int i=0; i<testData.size(); i++)
		{
			FeatureData& x = testData[i];
			int classId = classifier.Classify(x);
			x.Print();
			std::cout << " is classified as class " << classId << std::endl;
		}

		std::cout << std::endl;
	}

	Evaluator evaluator(trainingData);

	//Resubstitution Validation
	std::cout << "Begin Resubstitution Validation" << std::endl;
	evaluator.ResubstitutionValidate(bc);

	//Cross Validation
	std::cout << "Begin Cross Validation" << std::endl;
	evaluator.CrossValidate(bc, 4);
	std::cout << std::endl;

	testData.clear();
	delete &testData;

	return 0;
}