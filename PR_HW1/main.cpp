#include "BayesianClassifier.h"
#include <string>
#include <iostream>
#include <vector>
int main(int argc, char* argv[])
{
	BayesianClassifier bc;

	bc.ReadFile(std::string("data-iris.txt"));
	bc.PrintClassesInformation();
	float data[4] = {4.4,2.9,1.4,0.2};
	FeatureVector x = cvMat(4,1,CV_32FC1, data);

	std::vector<FeatureVector*> testData = bc.ReadTestData("testdata.txt");
	for(int i=0; i<testData.size(); i++)
	{
		FeatureVector* x = testData[i];
		Class& c = bc.Classfy(*x);
		std::cout << "("<< x->data.fl[0] << "," << x->data.fl[1] << "," << x->data.fl[2] << "," << x->data.fl[3]  << ") classfy to " << c.ID << std::endl;
		cvReleaseMat(&x);
	}
	return 0;
}