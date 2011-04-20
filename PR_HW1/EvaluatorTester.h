#ifndef _EVALUATOR_TESTER_H
#define _EVALUATOR_TESTER_H

#include <iostream>
#include "BayesianClassifier.h"
#include "NaiveBayesClassifier.h"
#include "Evaluator.h"
#include "TrainingData.h"
using namespace std;

class EvaluatorTester
{
public:
	static void Test(TrainingData& _trainingData, int run, int k, int n)
	{
		BayesianClassifier bc(_trainingData.NumberOfFeatures);
		NaiveBayesClassifier nc(_trainingData.NumberOfFeatures);
		Classifier* classifiers[2] = { &bc, &nc };

		Evaluator evaluator(_trainingData);
		evaluator.EnableOutput = false;

		vector<FeatureData*> trainingData;
		for(int i=0; i<_trainingData.Data.size() ; i++)
		{
			trainingData.push_back(&_trainingData.Data[i]);
		}
		
		TrainingData runTrainingData;
		float resubRates[2] = {0};
		float crossRates[2] = {0};
		srand(time(0));
		for(int runIdx=0; runIdx < run; runIdx++)
		{
			if(n != _trainingData.Data.size())
			{
				// shuffle training data
				for(int i=0; i<trainingData.size() ; i++)
				{
					int r = i + (rand() % (trainingData.size()-i));
					FeatureData* temp = trainingData[i];
					trainingData[i] = trainingData[r];
					trainingData[r] = temp;
				}
			}
			vector<FeatureData*> runTrainingFeatures;
			for(int i=0; i<n ; i++)
			{
				runTrainingFeatures.push_back(trainingData[i]);
			}

			runTrainingData = TrainingData(&runTrainingFeatures[0], n);
			evaluator.SetTrainingData(runTrainingData);

			for(int ci=0; ci<2; ci++)
			{
				//Resubstitution Validation
				//cout << "1. Resubstitution Validation" <<endl;
				resubRates[ci] += evaluator.ResubstitutionValidate(*classifiers[ci]);
				//cout <<endl;
				//Cross Validation
				//cout << "2. Begin Cross Validation" <<endl;
				crossRates[ci] += evaluator.CrossValidate(*classifiers[ci], k);
				//cout <<endl;
			}
		}

		cout << "===Evaluator Test===" << endl;
		cout << "run=" << run << endl;
		cout << "k=" << k << endl;
		cout << "n=" << n << endl;

		cout << "===Test Results===" << endl;
		for(int ci=0; ci<2; ci++)
		{
			resubRates[ci] /= run;
			crossRates[ci] /= run;

			cout.precision(4);
			cout << ci+1 << ". " << classifiers[ci]->GetName() << endl;
			cout << " Resubstitution: " << resubRates[ci] << endl;
			cout << " K-fold: " << crossRates[ci] << endl;
		}
	}
};
#endif