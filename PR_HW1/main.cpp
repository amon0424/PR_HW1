#include "BayesianClassifier.h"
#include "NaiveBayesClassifier.h"
#include "Evaluator.h"

#include <string>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
	Evaluator evaluator;
	

	evaluator.ReadFile(std::string("data-iris.txt"));
	std::vector<FeatureData> testData = evaluator.ReadTestData("testdata.txt");
	
	BayesianClassifier bc(evaluator.NumberOfFeatures);
	evaluator.InitializeClassifier(bc);
	evaluator.Train(bc);

	NaiveBayesClassifier nc(evaluator.NumberOfFeatures);
	evaluator.InitializeClassifier(nc);
	evaluator.Train(nc);
	
	// Bayesian Classifier
	std::cout << "Using Bayesian Classifier" << std::endl;
	int correct = 0;
	for(int i=0; i<testData.size(); i++)
	{
		FeatureData& x = testData[i];
		int classId = evaluator.Classify(bc, x);
		if(classId == x.ClassID)
		{
			correct++;
		}
		else
		{
			std::cout << "("<< x.FeatureVector->data.fl[0] << "," << x.FeatureVector->data.fl[1] << "," << x.FeatureVector->data.fl[2] << "," << x.FeatureVector->data.fl[3]  << ") classfy to " << classId;
			std::cout << ", wrong. correct is " << x.ClassID  << std::endl;
		}
	}
	std::cout << "Correct: " << correct << "/" << testData.size();
	printf("  %.2f%%\n", correct * 100.0 / testData.size());

	// Naive Bayes Classifier
	std::cout << "Using Naive Bayes Classifier" << std::endl;
	correct = 0;
	for(int i=0; i<testData.size(); i++)
	{
		FeatureData& x = testData[i];
		int classId = evaluator.Classify(nc, x);
		if(classId == x.ClassID)
		{
			correct++;
		}
		else
		{
			std::cout << "("<< x.FeatureVector->data.fl[0] << "," << x.FeatureVector->data.fl[1] << "," << x.FeatureVector->data.fl[2] << "," << x.FeatureVector->data.fl[3]  << ") classfy to " << classId;
			std::cout << ", wrong. correct is " << x.ClassID  << std::endl;
		}

		x.Delete();
	}
	std::cout << "Correct: " << correct << "/" << testData.size();
	printf("  %.2f%%\n", correct * 100.0 / testData.size());

	evaluator.Delete();
	return 0;
}