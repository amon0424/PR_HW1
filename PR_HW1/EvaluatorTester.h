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
	static void Test(TrainingData& ttlTrainingData, int run, int k, int n)
	{
		BayesianClassifier bc(ttlTrainingData.NumberOfFeatures);
		NaiveBayesClassifier nc(ttlTrainingData.NumberOfFeatures);
		Classifier* classifiers[2] = { &bc, &nc };

		Evaluator evaluator(ttlTrainingData);
		evaluator.EnableOutput = false;

		float resubRates[2] = {0};
		float crossRates[2] = {0};
		srand(time(0));
		for(int runIdx=0; runIdx < run; runIdx++)
		{
			TrainingData runTrainingData = ttlTrainingData.PickRandomData(n);
			evaluator.SetTrainingData(runTrainingData);

			for(int ci=0; ci<2; ci++)
			{
				//Resubstitution Validation
				resubRates[ci] += evaluator.ResubstitutionValidate(*classifiers[ci]);
				//Cross Validation
				crossRates[ci] += evaluator.CrossValidate(*classifiers[ci], k);
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