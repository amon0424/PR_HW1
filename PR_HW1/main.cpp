#include "BayesianClassifier.h"
#include <string>
int main(int argc, char* argv[])
{
	BayesianClassifier bc;

	bc.ReadFile(std::string("data-iris.txt"));

	return 0;
}