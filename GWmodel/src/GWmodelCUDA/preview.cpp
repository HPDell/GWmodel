#include "preview.h"
#include <stdio.h>

#include <cuda_runtime.h>
#include "helper.h"

void previewCudaMat(double *mat, size_t row, size_t col, const char* name)
{
	printf("%s\n", name);
	double* data = new double[row*col];
	cudaMemcpy(data, mat, sizeof(double) * row * col, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < row; i++)
	{
		if (i == row - 1)
		{
			if (i > 10) printf("\t|........\n");
			printf("\t|");
			for (size_t j = 0; j < col; j++)
			{
				if (j == col - 1)
				{
					if (j > 4) printf("%12s", "...");
					printf(MATRIX_PREVIEW_FORMAT, data[j * row + i]);
				}
				else if (j < 4)
				{
					printf(MATRIX_PREVIEW_FORMAT, data[j * row + i]);
				}
			}
			printf("\n");
		}
		else if (i < 10)
		{
			printf("\t|");
			for (size_t j = 0; j < col; j++)
			{
				if (j == col - 1)
				{
					if (j > 4) printf("%12s", "...");
					printf(MATRIX_PREVIEW_FORMAT, data[j * row + i]);
				}
				else if (j < 4)
				{
					printf(MATRIX_PREVIEW_FORMAT, data[j * row + i]);
				}
			}
			printf("\n");
		}
	}
	delete[] data;
}

void previewArmaMat(mat x, const char* name)
{
	printf("%s\n", name);
	int row = x.n_rows, col = x.n_cols;
	for (size_t i = 0; i < row; i++)
	{
		if (i < 10)
		{
			printf("|");
			for (size_t j = 0; j < col; j++)
			{
				if (j == col - 1)
				{
					printf("%12s", "...");
					printf(MATRIX_PREVIEW_FORMAT, x(i, j));
				}
				else if (j < 5)
				{
					printf(MATRIX_PREVIEW_FORMAT, x(i, j));
				}
			}
			printf("\n");
		}
		else if (i == row - 1)
		{
			if (i > 10) printf("|........\n");
			printf("|");
			for (size_t j = 0; j < col; j++)
			{
				if (j == col - 1)
				{
					printf("%12s", "...");
					printf(MATRIX_PREVIEW_FORMAT, x(i, j));
				}
				else if (j < 5)
				{
					printf(MATRIX_PREVIEW_FORMAT, x(i, j));
				}
			}
			printf("\n");
		}
	}
}

void previewInvInfo(const int *d_info, size_t n, const char* name)
{
	printf("%s\n", name);
	int* data = new int[n];
	cudaMemcpy(data, d_info, sizeof(int) * n, cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < n; i++)
	{
		if (i == n - 1)
		{
			if (i > 10) printf("...");
			printf("%8d ", data[i]);
			printf("\n");
		}
		else if (i < 10)
		{
			printf("%8d", data[i]);
		}
	}
	delete[] data;
}

void previewInvInfo(const int *d_info, size_t n, size_t begin, size_t end, const char* name)
{
	printf("%s\n", name);
	int* data = new int[n];
	cudaMemcpy(data, d_info, sizeof(int) * n, cudaMemcpyDeviceToHost);
	for (size_t i = begin; i < end; i++)
	{
		if (i == end - 1)
		{
			if (i > 10) printf("...");
			printf("%8d ", data[i]);
			printf("\n");
		}
		else if (i < 10)
		{
			printf("%8d ", data[i]);
		}
	}
	delete[] data;
}