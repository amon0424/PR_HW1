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
			printf("|");
			for(int j=0;j<cols;j++)
			{
				printf("%.1f\t", cvmGet(matrix, i, j));
			}
			printf("|\n");
		}
	}

	static void PrintIntMatrix(CvMat* matrix, int rows, int cols)
	{
		for(int i=0;i<rows;i++)
		{
			printf("|");
			for(int j=0;j<cols;j++)
			{
				
				printf("%.0f\t", cvmGet(matrix, i, j));
			}
			printf("|\n");
		}
	}

	static void ZeroMatrix(CvMat* matrix, int rows, int cols)
	{
		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
				cvmSet(matrix, i, j, 0);
			}
		}
	}
};
#endif