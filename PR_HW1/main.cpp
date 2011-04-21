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
	bool enableTesting = false;
	int k = 4;
	int n = -1;	//N

	// arguments for test mode
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
								cout << "Error: " << argument.substr(j+2) << " is not a valid value for k." << endl;
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
								cout << "Error: " << argument.substr(j+2) << " is not a valid value for n." << endl;
								return 1;
							}
							j += 2 + argument.substr(j+2).length();
						}
						break;
					case 'r':
						if(argument.length() > j+2)
						{
							if(!ifstream(argument.substr(j+2).c_str()))
							{
								cout << "Error: Testing data file dosen't exists." << endl;
								return 1;
							}
							testingFilename = argument.substr(j+2);
							j += 2 + argument.substr(j+2).length();
						}
						break;
					case 't':
						if(argument.length() > j+2)
						{
							if((r = atoi(argument.substr(j+2).c_str()))==0)
							{
								cout << "Error: " << argument.substr(j+2) << " is not a valid value for r." << endl;
								return 1;
							}
							j += 2 + argument.substr(j+2).length();
							enableTesting = true;
						}
						break;
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

		trainingDataFilename = string(argv[argc-1]);

		if(!ifstream(trainingDataFilename.c_str()))
		{
			cout << "Error: Training data file dosen't exists." << endl;
			return 1;
		}
	}
	
	if(trainingDataFilename.compare("") == 0 )
	{
		cout << "Error: Must provide training data file." << endl;
		return 1;
	}

	TrainingData ttlTrainingData;
	ttlTrainingData.ReadFile(trainingDataFilename);

	if(n == -1)
		n = ttlTrainingData.Data.size();

	if(n > ttlTrainingData.Data.size())
	{
		cout << "Error: N is larger than the size of training data." << endl;
		return 1;
	}

	TrainingData trainingData = ttlTrainingData.PickRandomData(n);
	
	if(!testMode)
	{
		// Classifiers
		BayesianClassifier bc(ttlTrainingData.NumberOfFeatures);
		NaiveBayesClassifier nc(ttlTrainingData.NumberOfFeatures);

		Classifier* classifiers[2] = { &bc, &nc };

		if(enableTesting)
		{
			// Begin classify
			cout << "===Testing===" <<endl <<endl;
			vector<FeatureData>& testData = *ttlTrainingData.ReadTestData(testingFilename);
			double* probability = new double[ttlTrainingData.Classes.size()];
			for(int classifierID = 0; classifierID < 2; classifierID++)
			{
				Classifier& classifier = *classifiers[classifierID];

				cout << endl << classifierID+1 << ". Using " << classifier.GetName() << endl << endl;

				classifier.SetClasses(&trainingData.Classes[0], trainingData.Classes.size());
				classifier.Train(&trainingData.Data[0], trainingData.Data.size());

				if(printClassesParameters)
					classifier.Print();

				cout << "Classification Results" << endl;
				cout << "-------------------------------------------------" << endl;
				cout << left << setw(22) << "Feature Vector";
				cout << right << setw(5) << "Class"; 
				for(int j=0; j<trainingData.Classes.size(); j++)
				{
					//cout.width(5);
					cout << left << "   P(" << j+1 << ")";
				}
				cout << endl;
				cout << "-------------------------------------------------" << endl;
				for(int i=0; i<testData.size(); i++)
				{
					FeatureData& x = testData[i];
					int classId = classifier.Classify(x, probability);
					
					x.Print();
					cout << right << setw(5) << ttlTrainingData.Classes[classId-1].Label << "   " ;
					for(int j=0; j<trainingData.Classes.size(); j++)
					{
						cout << left << setw(7) << fixed  << setprecision(3)  << probability[j];
					}
					cout << endl;
				}
				cout << "-------------------------------------------------" << endl;

				cout <<endl;
			}
			delete[] probability;
			testData.clear();
			delete &testData;
		}
		if(enableEvaluation)
		{
			cout << "===Evaluation===" << endl << endl;

			Evaluator evaluator(trainingData);

			for(int ci=0; ci<2; ci++)
			{
				cout << ci+1 << ". " << (*classifiers[ci]).GetName() << endl;
				//Resubstitution Validation
				cout << "Resubstitution Validation" <<endl;
				evaluator.ResubstitutionValidate(*classifiers[ci]);
				cout << endl;
				//Cross Validation
				cout << "Cross Validation" <<endl;
				evaluator.CrossValidate(*classifiers[ci], k);
				cout << endl;
			}
		}

		
	}
	else
	{
		EvaluatorTester::Test(ttlTrainingData, r, k, n);
	}	
	
	return 0;
}