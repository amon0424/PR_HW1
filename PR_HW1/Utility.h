#ifndef _UTILITY_H
#define _UTILITY_H
#include <opencv/cv.h>
class Utility
{
public:
	static void PrintMatrix(CvMat* matrix, int rows, int cols)
	{
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
				printf("%.1f ", matrix->data.fl[i*cols+j]);
			}
			printf("\n");
		}
	}
};
#endif