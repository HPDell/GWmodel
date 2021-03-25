#pragma once
#include <armadillo>

#include "IGWmodelCUDA.h"
#include "GWmodelKernel.h"

using namespace arma;

class CGWmodelCUDA : public IGWmodelCUDA
{
private:
	typedef bool (CGWmodelCUDA::*RegressionFunction)(double, double, bool, double, int, bool, int, int);

public:
	CGWmodelCUDA();
	CGWmodelCUDA(int N, int K, bool rp_given, int n, bool dm_given);
	~CGWmodelCUDA();

	virtual void SetX(int i, int k, double value);
	virtual void SetY(int i, double value);
	virtual void SetDp(int i, double u, double v);
	virtual void SetRp(int i, double u, double v);
	virtual void SetDmat(int i, int j, double value);

	virtual double GetBetas(int i, int k);
	virtual double GetBetasSE(int i, int k);
	virtual double GetShat1();
	virtual double GetShat2();
	virtual double GetQdiag(int i);


	virtual bool Regression(
		bool hatmatrix, bool ftest,
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	);

	virtual double CV(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	);

	bool RegressionWithHatmatrixFtest(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	);

	bool RegressionWithHatmatrix(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	);

	bool RegressionOnly(
		double p, double theta, bool longlat,
		double bw, int kernel, bool adaptive,
		int groupl, int gpuID
	);
	
private:
	mat x;
	vec y;
	mat dp;
	mat rp;
	mat dMat;
	bool rp_given;
	bool dm_given;
	mat betas;
	mat betasSE;
	vec s_hat;
	vec qdiag;

	RegressionFunction mRegressionFunction = &CGWmodelCUDA::RegressionOnly;
};
