#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv/cv.h>

#include "BayesianClassifier.h"
#include "NaiveBayesClassifier.h"
#include "Evaluator.h"
#include "TrainingData.h"
#include "Utility.h"
#include "EvaluatorTester.h"

using namespace std;

int main(int argc, char* argv[])
{
	bool enableEvaluation = false;
	bool printClassesParameters = false;
	int k = 0;

	// Process the arguments
	string trainingDataFilename;
	string testingFilename;

	if(argc > 3)
	{
		// Options
		for(int i=1; i<argc-2; i++)
		{
			string argument(argv[i]);
			/*if(argument.compare(0,2,"--") == 0)
			{
				argument = string(argument.substr(2));
				if(argument.compare(0, 12, "testing-file") == 0)
				{
					testingFilename = argument.substr(12);
				}
			}
			else */if(argument[0] == '-')
			{
				for(int j=1; j<argument.length(); j++)
				{
					switch(argument[j])
					{
					case 'e':
						enableEvaluation = true;
						break;
					case 'i':
						printClassesParameters = true;
						break;
					}
				}
			}
		}

		// The last 3 arguments is training and testing file name.

		trainingDataFilename = string(argv[argc-2]);
		if(!ifstream(trainingDataFilename.c_str()))
		{
			cout << "Training data file dosen't exists." << endl;
			return 1;
		}

		testingFilename = string(argv[argc-1]);
		if(!ifstream(testingFilename.c_str()))
		{
			cout << "Testing data file dosen't exists." << endl;
			return 1;
		}
	}
	else
	{
		cout << "Must provide training and testing data file." << endl;
		return 1;
	}

	TrainingData trainingData;
	trainingData.ReadFile(trainingDataFilename);
	
	// Classifiers
	BayesianClassifier bc(trainingData.NumberOfFeatures);
	NaiveBayesClassifier nc(trainingData.NumberOfFeatures);

	Classifier* classifiers[2] = { &bc, &nc };

	// Begin classify
	cout << "===Testing===" <<endl <<endl;
	vector<FeatureData>& testData = *trainingData.ReadTestData(testingFilename);
	for(int classifierID = 0; classifierID < 2; classifierID++)
	{
		Classifier& classifier = *classifiers[classifierID];

		cout << endl << classifierID+1 << ". Using " << classifier.GetName() << endl << endl;

		classifier.SetClasses(&trainingData.Classes[0], trainingData.Classes.size());
		classifier.Train(&trainingData.Data[0], trainingData.Data.size());

		if(printClassesParameters)
			classifier.Print();

		cout << "Classification Results" << endl;
		cout << "---------------------------" << endl;
		cout << left << setw(22) << "Feature Vector";
		cout << right << setw(5) << "Class" <<endl;
		cout << "---------------------------" << endl;
		for(int i=0; i<testData.size(); i++)
		{
			FeatureData& x = testData[i];
			int classId = classifier.Classify(x);
			x.Print();
			cout << right << setw(5) << classId <<endl;
		}
		cout << "---------------------------" << endl;

		cout <<endl;
	}

	if(enableEvaluation)
	{
		cout << "===Evaluation===" <<endl <<endl;

		Evaluator evaluator(trainingData);

		//Resubstitution Validation
		cout << "1. Resubstitution Validation" <<endl;
		evaluator.ResubstitutionValidate(bc);
		cout <<endl;

		//Cross Validation
		cout << "2. Begin Cross Validation" <<endl;
		evaluator.CrossValidate(bc, 4);
		cout <<endl;
	}

	EvaluatorTester::Test(trainingData, 10, 4, 150);

	testData.clear();
	delete &testData;
	
	return 0;
}