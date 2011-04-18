#include <string>
#include <iostream>
#include <vector>
#include <opencv/cv.h>

#include "BayesianClassifier.h"
#include "NaiveBayesClassifier.h"
#include "Evaluator.h"
#include "Utility.h"

int main(int argc, char* argv[])
{
	Evaluator evaluator;
	evaluator.ReadFile(std::string("data-iris.txt"));
	std::vector<FeatureData> testData = evaluator.ReadTestData("testdata.txt");
	
	// Classifiers
	BayesianClassifier bc(evaluator.NumberOfFeatures);
	NaiveBayesClassifier nc(evaluator.NumberOfFeatures);

	Classifier* classifiers[2] = { &bc, &nc };

	// Begin classify
	CvMat* confusionMat = cvCreateMat(evaluator.NumberOfClasses, evaluator.NumberOfClasses, CV_32FC1);
	for(int classifierID = 0; classifierID < 2; classifierID++)
	{
		Classifier& classifier = *classifiers[classifierID];
		evaluator.InitializeClassifier(classifier);
		evaluator.Train(classifier);

		std::cout << "Using " << classifier.GetName() << std::endl;

		Utility::ZeroMatrix(confusionMat, evaluator.NumberOfClasses, evaluator.NumberOfClasses);
		int correct = 0;

		for(int i=0; i<testData.size(); i++)
		{
			FeatureData& x = testData[i];
			int classId = evaluator.Classify(classifier, x);
			if(classId == x.ClassID)
			{
				correct++;
				cvmSet(confusionMat, classId-1, classId-1, cvmGet(confusionMat, classId-1, classId-1) + 1);
			}
			else
			{
				std::cout << "("<< x.FeatureVector->data.fl[0] << "," << x.FeatureVector->data.fl[1] << "," << x.FeatureVector->data.fl[2] << "," << x.FeatureVector->data.fl[3]  << ") classfy to " << classId;
				std::cout << ", wrong. correct is " << x.ClassID  << std::endl;
				cvmSet(confusionMat, x.ClassID-1, classId-1, cvmGet(confusionMat, x.ClassID-1, classId-1) + 1);
			}
		}
		std::cout << "Correct: " << correct << "/" << testData.size();
		printf("  %.2f%%\n", correct * 100.0 / testData.size());

		std::cout << "Confusion Matrix: " << std::endl;
		Utility::PrintIntMatrix(confusionMat, evaluator.NumberOfClasses, evaluator.NumberOfClasses);

		std::cout << std::endl;
	}
	cvReleaseMat(&confusionMat);

	for(int i=0; i<testData.size(); i++)
			testData[i].Delete();

	evaluator.Delete();
	return 0;
}