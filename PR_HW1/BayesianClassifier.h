#include <string>
#include <fstream>
#include <vector>
#include "Class.h"

using std::string;

typedef std::vector<float> FeatureVector;

class BayesianClassifier 
{
	std::vector<FeatureVector> _tranningData;
	std::vector<Class> _classes;

	int _numberOfClasses;
	int _numberOfFeatures;

public:
	BayesianClassifier()
	{
		
	}

	void ReadFile(string filename)
	{
		std::ifstream inputFile(filename.c_str());

		inputFile >> _numberOfClasses;
		inputFile >> _numberOfFeatures;

		for(int i=0; i < _numberOfClasses; i++)
			_classes.push_back(Class(_numberOfFeatures));

		int tmp;
		inputFile >> tmp >> tmp >> tmp;

		while(!inputFile.eof())
		{
			try
			{
				FeatureVector featureVector;
				for(int i=0; i<_numberOfFeatures; i++)
				{
					float value;
					inputFile >> value;
					featureVector.push_back(value);
				}
				int classID;
				inputFile >> classID;

				_classes[classID-1].AddTranningData(featureVector);
			}
			catch (std::exception e) 
			{
			}
		}

		for(int i=0;i<_numberOfClasses; i++)
		{
			_classes[i].ComputeParamaters();
		}
	}

	void AddTranningData(FeatureVector& featureVector, int classID)
	{
	}
};
