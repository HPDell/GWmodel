#pragma once

#include <cuda_runtime.h>

typedef cudaError_t (*GW_DIST_FUNC)(double *d_dp, double *d_rp, int ndp, int nrp, int focus, double p, double *d_dists, int threads);
cudaError_t gw_dist_cuda_sp(double *d_dp, double *d_rp, int ndp, int nrp, int focus, double p, double *d_dists, int threads);
cudaError_t gw_dist_cuda_eu(double *d_dp, double *d_rp, int ndp, int nrp, int focus, double p, double *d_dists, int threads);
cudaError_t gw_dist_cuda_cd(double *d_dp, double *d_rp, int ndp, int nrp, int focus, double p, double *d_dists, int threads);
cudaError_t gw_dist_cuda_md(double *d_dp, double *d_rp, int ndp, int nrp, int focus, double p, double *d_dists, int threads);
cudaError_t gw_dist_cuda_mk(double *d_dp, double *d_rp, int ndp, int nrp, int focus, double p, double *d_dists, int threads);
GW_DIST_FUNC gw_dist_cuda(bool longlat, double p);

cudaError_t gw_coordinate_rotate_cuda(double* d_coords, int n, double theta, int threads);
cudaError_t gw_weight_cuda(double bw, int kernel, bool adaptive, double *d_dists, double *d_weight, int ndp, int nrp, int threads);
cudaError_t gw_xtw_cuda(const double* d_x, const double* d_weight, int n, int k, double* d_xtw, int threads);
cudaError_t gw_xdy_cuda(const double* d_x, const double* d_y, int n, double * d_xdoty, int threads);
cudaError_t gw_xdx_cuda(const double* d_x, int n, double * d_xdotx, int threads);
cudaError_t gw_qdiag_cuda(const double* d_si, int n, int p, double* d_q, int threads);
