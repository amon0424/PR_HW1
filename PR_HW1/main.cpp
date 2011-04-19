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
	bool testMode = false;
	int k = 4;

	// arguments for test mode
	int n = -1;	//N
	int r = 10;	//run

	// Process the arguments
	string trainingDataFilename;
	string testingFilename;

	if(argc > 2)
	{
		// Options
		for(int i=1; i<argc; i++)
		{
			string argument(argv[i]);
			if(argument.compare(0,2,"--") == 0)
			{
				argument = string(argument.substr(2));
				if(argument.compare(0, 4, "test") == 0)
				{
					testMode = true;
				}
			}
			else if(argument[0] == '-')
			{
				for(int j=1; j<argument.length(); j++)
				{
					switch(argument[j])
					{
					case 'k':
						if(argument.length() > j+2)
						{
							if((k = atoi(argument.substr(j+2).c_str()))==0)
							{
								cout << argument.substr(j+2) << " is not a valid value for k." << endl;
								return 1;
							}
							j += 2 + argument.substr(j+2).length();
						}
						break;
					case 'n':
						if(argument.length() > j+2)
						{
							if((n = atoi(argument.substr(j+2).c_str()))==0)
							{
								cout << argument.substr(j+2) << " is not a valid value for d." << endl;
								return 1;
							}
							j += 2 + argument.substr(j+2).length();
						}
					case 'r':
						if(argument.length() > j+2)
						{
							if((r = atoi(argument.substr(j+2).c_str()))==0)
							{
								cout << argument.substr(j+2) << " is not a valid value for r." << endl;
								return 1;
							}
							j += 2 + argument.substr(j+2).length();
						}
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

		if(!testMode)
		{
			trainingDataFilename = string(argv[argc-2]);
			testingFilename = string(argv[argc-1]);
		}
		else
		{
			trainingDataFilename = string(argv[argc-1]);
		}

		if(!ifstream(trainingDataFilename.c_str()))
		{
			cout << "Training data file dosen't exists." << endl;
			return 1;
		}
		if(!testMode && !ifstream(testingFilename.c_str()))
		{
			cout << "Testing data file dosen't exists." << endl;
			return 1;
		}
	}
	
	if(trainingDataFilename.compare("") == 0 || (!testMode && testingFilename.compare("") == 0))
	{
		cout << "Must provide training and testing data file." << endl;
		return 1;
	}

	TrainingData trainingData;
	trainingData.ReadFile(trainingDataFilename);

	if(n == -1)
		n = trainingData.Data.size();
	
	if(!testMode)
	{
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

			for(int ci=0; ci<2; ci++)
			{
				cout << ci << ". " << (*classifiers[ci]).GetName() << endl;
				//Resubstitution Validation
				cout << "Resubstitution Validation" <<endl;
				evaluator.ResubstitutionValidate(*classifiers[ci]);
				cout << endl;
				//Cross Validation
				cout << "Begin Cross Validation" <<endl;
				evaluator.CrossValidate(*classifiers[ci], k);
				cout << endl;
			}
		}

		testData.clear();
		delete &testData;
	}
	else
	{
		EvaluatorTester::Test(trainingData, r, k, n);
	}	
	
	return 0;
}