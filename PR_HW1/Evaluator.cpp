#include "Evaluator.h"
#include "TrainingData.h"
using namespace std;
float Evaluator::CrossValidate(Classifier& classifier, int k)
{
	// copy training data
	vector<FeatureData*> trainingData;
	for(int i=0; i<_trainingData->Data.size() ; i++)
	{
		trainingData.push_back(&_trainingData->Data[i]);
	}

	// shuffle training data
	srand(time(0));
	for(int i=0; i<trainingData.size() ; i++)
	{
		int r = i + (rand() % (trainingData.size()-i));
		FeatureData* temp = trainingData[i];
		trainingData[i] = trainingData[r];
		trainingData[r] = temp;
	}

	// divide
	vector<vector<FeatureData>> subsets;
	int numberOfSubset = trainingData.size() / k;
	int remain = trainingData.size() - numberOfSubset * k;

	vector<FeatureData*>::iterator current = trainingData.begin();
	for(int i=0; i<k; i++)
	{
		vector<FeatureData> subset;
		for(int j=0; j<numberOfSubset; j++)
		{
			subset.push_back(**current);
			current++;
		}
		if(remain > 0)
		{
			subset.push_back(**current);
			current++;
			remain--;
		}

		subsets.push_back(subset);
	}

	// begin validation
	CvMat* confusionMat = cvCreateMat(_trainingData->NumberOfClasses, _trainingData->NumberOfClasses, CV_32FC1);
	Utility::ZeroMatrix(confusionMat, _trainingData->NumberOfClasses, _trainingData->NumberOfClasses);
	int correct = 0;
	std::string hozLine;
	for(int i=0; i< 5 * this->_trainingData->NumberOfFeatures + 20 + 7*_trainingData->Classes.size(); i++)
		hozLine += "-";

	if(this->EnableOutput)
	{
		cout << "Incorrect Classification Results" << endl;
		cout << hozLine << endl;
		cout << left << setw(5 * this->_trainingData->NumberOfFeatures + 2) << "Feature Vector";
		cout << right << setw(9) << "Incorrect";
		cout << right << setw(9) << "Correct";
		for(int j=0; j<_trainingData->Classes.size(); j++)
		{
			cout << left << "   P(" <<  _trainingData->Classes[j].Label << ")";
		}
		cout << endl;
		cout << hozLine << endl;
	}
	for(int i=0; i<k; i++)
	{
		classifier.SetClasses(&_trainingData->Classes[0], _trainingData->Classes.size());

		//training
		for(int j=(i+1)%k; j!=i; j=(j+1)%k)
		{
			vector<FeatureData> estimationSubset = subsets[j];
			classifier.Train(&estimationSubset[0], estimationSubset.size());
		}

		//testing
		vector<FeatureData> testingSubset = subsets[i];
		double* probability = new double[_trainingData->Classes.size()];
		for(int j=0; j<testingSubset.size(); j++)
		{
			FeatureData& x = testingSubset[j];
			int classId = classifier.Classify(x, probability);

			if(classId == x.ClassID)
			{
				correct++;
				cvmSet(confusionMat, classId-1, classId-1, cvmGet(confusionMat, classId-1, classId-1) + 1);
			}
			else
			{
				if(this->EnableOutput)
				{
					x.Print();
					cout << right << setw(9) << _trainingData->Classes[classId-1].Label ;
					cout << right << setw(9) << _trainingData->Classes[x.ClassID-1].Label;
					for(int k=0; k<_trainingData->Classes.size(); k++)
					{
						cout << right << setw(7) << fixed  << setprecision(3)  << probability[k];
					}
					cout << endl;
				}
				cvmSet(confusionMat, x.ClassID-1, classId-1, cvmGet(confusionMat, x.ClassID-1, classId-1) + 1);
			}
		}
		delete[] probability;
	}
	if(this->EnableOutput)
	{
		cout << hozLine << endl;

		cout << "Correct: " << correct << "/" << trainingData.size();
		printf("  %.2f%%\n", correct * 100.0 / trainingData.size());

		cout << "Confusion Matrix: " << endl;
		Utility::PrintIntMatrix(confusionMat, _trainingData->NumberOfClasses, _trainingData->NumberOfClasses);
	}

	cvReleaseMat(&confusionMat);

	return (float)correct / trainingData.size();
}
float Evaluator::ResubstitutionValidate(Classifier& classifier)
{
	classifier.SetClasses(&_trainingData->Classes[0], _trainingData->Classes.size());
	classifier.Train(&_trainingData->Data[0], _trainingData->Data.size());

	// begin validation
	int correct = 0;
	CvMat* confusionMat = cvCreateMat(_trainingData->NumberOfClasses, _trainingData->NumberOfClasses, CV_32FC1);
	Utility::ZeroMatrix(confusionMat, _trainingData->NumberOfClasses, _trainingData->NumberOfClasses);

	std::string hozLine;
	for(int i=0; i< 5 * this->_trainingData->NumberOfFeatures + 20 + 7*_trainingData->Classes.size(); i++)
		hozLine += "-";

	if(this->EnableOutput)
	{
		cout << "Incorrect Classification Results" << endl;
		cout << hozLine << endl;
		cout << left << setw(5 * this->_trainingData->NumberOfFeatures + 2) << "Feature Vector";
		cout << right << setw(9) << "Incorrect";
		cout << right << setw(9) << "Correct";
		for(int j=0; j<_trainingData->Classes.size(); j++)
		{
			cout << left << "   P(" << _trainingData->Classes[j].Label << ")";
		}
		cout<< endl;
		cout << hozLine << endl;
	}
	double* probability = new double[_trainingData->Classes.size()];
	for(int j=0; j<_trainingData->Data.size(); j++)
	{
		FeatureData& x = _trainingData->Data[j];
		int classId = classifier.Classify(x, probability);

		if(classId == x.ClassID)
		{
			correct++;
			cvmSet(confusionMat, classId-1, classId-1, cvmGet(confusionMat, classId-1, classId-1) + 1);
		}
		else
		{
			if(this->EnableOutput)
			{
				x.Print();
				cout << right << setw(9) << _trainingData->Classes[classId-1].Label ;
				cout << right << setw(9) <<_trainingData->Classes[x.ClassID-1].Label;
				for(int k=0; k<_trainingData->Classes.size(); k++)
				{
					cout << right << setw(7) << fixed  << setprecision(3)  << probability[k];
				}
				cout << endl;
			}
			cvmSet(confusionMat, x.ClassID-1, classId-1, cvmGet(confusionMat, x.ClassID-1, classId-1) + 1);
		}
	}
	delete[] probability;
	if(this->EnableOutput)
	{
		cout << hozLine << endl;
		cout << "Correct: " << correct << "/" << _trainingData->Data.size();
		printf("  %.2f%%\n", correct * 100.0 / _trainingData->Data.size());

		cout << "Confusion Matrix: " << endl;
		Utility::PrintIntMatrix(confusionMat, _trainingData->NumberOfClasses, _trainingData->NumberOfClasses);
	}
	cvReleaseMat(&confusionMat);

	return (float)correct / _trainingData->Data.size();
}