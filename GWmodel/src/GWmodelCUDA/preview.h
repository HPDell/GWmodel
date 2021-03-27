#pragma once

#include <armadillo>

#define MATRIX_PREVIEW_FORMAT "%16.6lf"

using namespace arma;

void previewCudaMat(double *mat, size_t row, size_t col, const char* name);
void previewArmaMat(mat x, const char* name);

void previewInvInfo(const int *d_info, size_t n, const char* name);
void previewInvInfo(const int *d_info, size_t n, size_t begin, size_t end, const char* name);