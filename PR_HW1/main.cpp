#include "BayesianClassifier.h"
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
	
	int correct = 0;
	for(int i=0; i<testData.size(); i++)
	{
		FeatureData& x = testData[i];
		int classId = evaluator.Classify(bc, x);
		std::cout << "("<< x.FeatureVector->data.fl[0] << "," << x.FeatureVector->data.fl[1] << "," << x.FeatureVector->data.fl[2] << "," << x.FeatureVector->data.fl[3]  << ") classfy to " << classId;
		if(classId == x.ClassID)
		{
			std::cout << " correct." << std::endl;
			correct++;
		}
		else
			std::cout << " wrond. " << x.ClassID  << std::endl;

		x.Delete();
	}

	std::cout << "Correct: " << correct << "/" << testData.size() << std::endl;
	evaluator.Delete();
	return 0;
}