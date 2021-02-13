// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>

#ifdef CUDA_ACCE
#include <IGWmodelCUDA.h>
#endif

using namespace Rcpp;
using namespace arma;


//GWmodel_gw_dist
arma::mat gw_dist(arma::mat dp, arma::mat rp, int focus, double p, double theta, bool longlat, bool rp_given);
RcppExport SEXP GWmodel_gw_dist(SEXP dpSEXP, SEXP rpSEXP, SEXP focusSEXP, SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP rp_givenSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type dp(dpSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type rp(rpSEXP);
  Rcpp::traits::input_parameter< int >::type focus(focusSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< bool >::type rp_given(rp_givenSEXP);
  __result = Rcpp::wrap(gw_dist(dp, rp, focus, p, theta, longlat, rp_given));
  return __result;
  END_RCPP
}

// GWmodel_gw_weight
arma::mat gw_weight(arma::mat dist, double bw, int kernel, bool adaptive);
RcppExport SEXP GWmodel_gw_weight(SEXP distSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type dist(distSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< int >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(gw_weight(dist, bw, kernel, adaptive));
    return rcpp_result_gen;
END_RCPP
}

// GWmodel_gw_weight_vec
arma::vec gw_weight_vec(arma::vec vdist, double bw, std::string kernel, bool adaptive);
RcppExport SEXP GWmodel_gw_weight_vec(SEXP vdistSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type vdist(vdistSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(gw_weight_vec(vdist, bw, kernel, adaptive));
    return rcpp_result_gen;
END_RCPP
}

// GWmodel_gw_weight_mat
arma::mat gw_weight_mat(arma::mat mdist, double bw, std::string kernel, bool adaptive);
RcppExport SEXP GWmodel_gw_weight_mat(SEXP mdistSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type mdist(mdistSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(gw_weight_mat(mdist, bw, kernel, adaptive));
    return rcpp_result_gen;
END_RCPP
}
//GWmodel_gw_reg
Rcpp::List gw_reg(mat x, vec y, vec w, bool hatmatrix, int focus);
RcppExport SEXP GWmodel_gw_reg(SEXP xSEXP, SEXP ySEXP, SEXP wSEXP, SEXP hatmatrixSEXP,SEXP focusSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
	Rcpp::traits::input_parameter< bool >::type hatmatrix(hatmatrixSEXP);
	Rcpp::traits::input_parameter< int >::type focus(focusSEXP);
    __result = Rcpp::wrap(gw_reg(x, y, w, hatmatrix, focus));
    return __result;
END_RCPP
}

//GWmodel_gw_reg_1
Rcpp::List gw_reg_1(mat x, vec y, vec w);
RcppExport SEXP GWmodel_gw_reg_1(SEXP xSEXP, SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    __result = Rcpp::wrap(gw_reg_1(x, y, w));
    return __result;
END_RCPP
}
// trhat2
arma::vec trhat2(arma::mat S);
RcppExport SEXP GWmodel_trhat2(SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type S(SSEXP);
    rcpp_result_gen = Rcpp::wrap(trhat2(S));
    return rcpp_result_gen;
END_RCPP
}
//GWmodel_fitted
arma::vec fitted(arma::mat X, arma::mat beta);
RcppExport SEXP GWmodel_fitted(SEXP XSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
	Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    __result = Rcpp::wrap(fitted(X, beta));
    return __result;
END_RCPP
}

//GWmodel_ehat
arma::vec ehat(arma::vec y, arma::mat X, arma::mat beta);
RcppExport SEXP GWmodel_ehat(SEXP ySEXP, SEXP XSEXP, SEXP betaSEXP) {
BEGIN_RCPP
  Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    __result = Rcpp::wrap(ehat(y, X, beta));
    return __result;
END_RCPP
}

//GWmodel_rss
double rss(arma::vec y, arma::mat X, arma::mat beta);
RcppExport SEXP GWmodel_rss(SEXP ySEXP, SEXP XSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    __result = Rcpp::wrap(rss(y, X, beta));
    return __result;
END_RCPP
}

//GWmodel_gwr_diag
arma::vec gwr_diag(arma::vec y,arma::mat x, arma::mat beta, arma::mat S);
RcppExport SEXP GWmodel_gwr_diag(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type S(SSEXP);
    __result = Rcpp::wrap(gwr_diag(y, x, beta, S));
    return __result;
END_RCPP
}

//GWmodel_gwr_diag1
arma::vec gwr_diag1(arma::vec y,arma::mat x, arma::mat beta, arma::vec s_hat);
RcppExport SEXP GWmodel_gwr_diag1(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP s_hatSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
	Rcpp::traits::input_parameter< arma::vec >::type s_hat(s_hatSEXP);
    __result = Rcpp::wrap(gwr_diag1(y, x, beta, s_hat));
    return __result;
END_RCPP
}
//GWmodel_AICc
double AICc(arma::vec y,arma::mat x, arma::mat beta, arma::mat S);
RcppExport SEXP GWmodel_AICc(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type S(SSEXP);
    __result = Rcpp::wrap(AICc(y, x, beta, S));
    return __result;
END_RCPP
}

//GWmodel_AICc1
double AICc1(arma::vec y,arma::mat x, arma::mat beta, arma::vec s_hat);
RcppExport SEXP GWmodel_AICc1(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP s_hatSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
	Rcpp::traits::input_parameter< arma::vec >::type s_hat(s_hatSEXP);
    __result = Rcpp::wrap(AICc1(y, x, beta, s_hat));
    return __result;
END_RCPP
}

//GWmodel_AICc_rss
arma::vec AICc_rss(arma::vec y,arma::mat x, arma::mat beta, arma::mat S);
RcppExport SEXP GWmodel_AICc_rss(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP SSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
	Rcpp::traits::input_parameter< arma::mat >::type S(SSEXP);
    __result = Rcpp::wrap(AICc_rss(y, x, beta, S));
    return __result;
END_RCPP
}

//GWmodel_AICc_rss1
arma::vec AICc_rss1(arma::vec y,arma::mat x, arma::mat beta, arma::vec s_hat);
RcppExport SEXP GWmodel_AICc_rss1(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP s_hatSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type s_hat(s_hatSEXP);
  __result = Rcpp::wrap(AICc_rss1(y, x, beta, s_hat));
  return __result;
  END_RCPP
}

//GWmodel_Ci_mat
arma::mat Ci_mat(arma::mat x, arma::vec w);
RcppExport SEXP GWmodel_Ci_mat(SEXP xSEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
	Rcpp::RNGScope __rngScope;
	Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
	Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    __result = Rcpp::wrap(Ci_mat(x,w));
    return __result;
END_RCPP
}

//GWmodel_gw_local_r2
arma::vec gw_local_r2(mat dp, vec dybar2, vec dyhat2, bool dm_given, mat dmat, double p, double theta, bool longlat, double bw, int kernel, bool adaptive);
RcppExport SEXP GWmodel_gw_local_r2(SEXP dpSEXP, SEXP dybar2SEXP, SEXP dyhat2SEXP, SEXP dm_givenSEXP, SEXP dmatSEXP,
                                    SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type dp(dpSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type dybar2(dybar2SEXP);
  Rcpp::traits::input_parameter< arma::vec >::type dyhat2(dyhat2SEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dmat(dmatSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  __result = Rcpp::wrap(gw_local_r2(dp, dybar2, dyhat2, dm_given, dmat, p, theta, longlat, bw, kernel, adaptive));
  return __result;
  END_RCPP
}

// GWmodel_BIC
double BIC(arma::vec y, arma::mat x, arma::mat beta, arma::vec s_hat);
RcppExport SEXP GWmodel_BIC(SEXP ySEXP, SEXP xSEXP, SEXP betaSEXP, SEXP s_hatSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type s_hat(s_hatSEXP);
    rcpp_result_gen = Rcpp::wrap(BIC(y, x, beta, s_hat));
    return rcpp_result_gen;
END_RCPP
}

// GWmodel_gw_reg_2
arma::mat gw_reg_2(arma::mat x, arma::vec y, arma::vec w);
RcppExport SEXP GWmodel_gw_reg_2(SEXP xSEXP, SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(gw_reg_2(x, y, w));
    return rcpp_result_gen;
END_RCPP
}

// GWmodel_gwr_q
arma::mat gwr_q(arma::mat x, arma::vec y, arma::mat dMat, double bw, std::string kernel, bool adaptive);
RcppExport SEXP GWmodel_gwr_q(SEXP xSEXP, SEXP ySEXP, SEXP dMatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dMat(dMatSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(gwr_q(x, y, dMat, bw, kernel, adaptive));
    return rcpp_result_gen;
END_RCPP
}

//GWmodel_scgwr_pre
Rcpp::List scgwr_pre(arma::mat x, arma::vec y, int bw, int poly, double b0, arma::mat g0, arma::mat neighbour);
RcppExport SEXP GWmodel_scgwr_pre(SEXP xSEXP, SEXP ySEXP, SEXP bwSEXP, SEXP polySEXP, SEXP b0SEXP, SEXP g0SEXP, SEXP neighbourSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< int >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< int >::type poly(polySEXP);
  Rcpp::traits::input_parameter< double >::type b0(b0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type g0(g0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type neighbour(neighbourSEXP);
  __result = Rcpp::wrap(scgwr_pre(x, y, bw, poly, b0, g0, neighbour));
  return __result;
  END_RCPP
}

//GWmodel_scgwr_loocv
double scgwr_loocv(arma::vec target, arma::mat x, arma::vec y, int bw, int poly, arma::mat Mx0, arma::mat My0, arma::mat XtX, arma::mat XtY);
RcppExport SEXP GWmodel_scgwr_loocv(SEXP targetSEXP, SEXP xSEXP, SEXP ySEXP, SEXP bwSEXP, SEXP polySEXP, SEXP Mx0SEXP, SEXP My0SEXP, SEXP XtXSEXP, SEXP XtYSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::vec >::type target(targetSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< int >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< int >::type poly(polySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Mx0(Mx0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type My0(My0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type XtX(XtXSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type XtY(XtYSEXP);
  __result = Rcpp::wrap(scgwr_loocv(target, x, y, bw, poly, Mx0, My0, XtX, XtY));
  return __result;
  END_RCPP
}

//GWmodel_scgwr_reg
Rcpp::List scgwr_reg(arma::mat x, arma::vec y, int bw, int poly, arma::mat G0, arma::mat Mx0, arma::mat My0, arma::mat XtX, arma::mat XtY, arma::mat neighbour, arma::vec parameters);
RcppExport SEXP GWmodel_scgwr_reg(SEXP xSEXP, SEXP ySEXP, SEXP bwSEXP, SEXP polySEXP, 
                                  SEXP G0SEXP, SEXP Mx0SEXP, SEXP My0SEXP, SEXP XtXSEXP, SEXP XtYSEXP, 
                                  SEXP neighbourSEXP, SEXP parametersSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< int >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< int >::type poly(polySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type G0(G0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type Mx0(Mx0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type My0(My0SEXP);
  Rcpp::traits::input_parameter< arma::mat >::type XtX(XtXSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type XtY(XtYSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type neighbour(neighbourSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type parameters(parametersSEXP);
  __result = Rcpp::wrap(scgwr_reg(x, y, bw, poly, G0, Mx0, My0, XtX, XtY, neighbour, parameters));
  return __result;
  END_RCPP
}

//GWmodel_gw_reg_all
Rcpp::List gw_reg_all(arma::mat x, arma::vec y, arma::mat dp, bool rp_given, arma::mat rp, bool dm_given, arma::mat dmat, bool hatmatrix, 
                      double p, double theta, bool longlat, 
                      double bw, int kernel, bool adaptive,
                      int ngroup, int igroup);
RcppExport SEXP GWmodel_gw_reg_all(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP rp_givenSEXP, SEXP rpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, SEXP hatmatrixSEXP, 
                                   SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                   SEXP ngroupSEXP, SEXP igroupSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dp(dpSEXP);
  Rcpp::traits::input_parameter< bool >::type rp_given(rp_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type rp(rpSEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dmat(dmatSEXP);
  Rcpp::traits::input_parameter< bool >::type hatmatrix(hatmatrixSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  Rcpp::traits::input_parameter< int >::type ngroup(ngroupSEXP);
  Rcpp::traits::input_parameter< int >::type igroup(igroupSEXP);
  __result = Rcpp::wrap(gw_reg_all(x, y, dp, rp_given, rp, dm_given, dmat, hatmatrix, p, theta, longlat, bw, kernel, adaptive, ngroup, igroup));
  return __result;
  END_RCPP
}

#ifdef CUDA_ACCE
RcppExport SEXP GWmodel_gw_reg_cuda(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP rp_givenSEXP, SEXP rpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, SEXP hatmatrixSEXP, 
                                   SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                   SEXP ngroupSEXP, SEXP gpuIDSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type xT(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type yT(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dpT(dpSEXP);
  Rcpp::traits::input_parameter< bool >::type rp_given(rp_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type rpT(rpSEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dMatT(dmatSEXP);
  Rcpp::traits::input_parameter< bool >::type hatmatrix(hatmatrixSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  Rcpp::traits::input_parameter< int >::type ngroup(ngroupSEXP);
  Rcpp::traits::input_parameter< int >::type gpuID(gpuIDSEXP);
  
  mat x = mat(xT);
  mat y = mat(yT);
  mat dp = mat(dpT);
  mat rp = mat(rpT);
  mat dMat = mat(dMatT);
  int N = x.n_rows;
  int K = x.n_cols;
  int n = rp.n_rows;
  IGWmodelCUDA* cuda = GWCUDA_Create(N, K, rp_given, n, dm_given);
  for (int r = 0; r < N; r++) {
    for (int c = 0; c < K; c++) {
      cuda->SetX(c, r, x(r, c));
    }
    cuda->SetY(r, y(r));
    cuda->SetDp(r, dp(r, 0), dp(r, 1));
  }
  if (rp_given) {
    for (int r = 0; r < n; r++) {
      cuda->SetRp(r, rp(r, 0), rp(r, 1));
    }
  }
  if (dm_given) {
    for (int d = 0; d < N; d++) {
      for (int r = 0; r < n; r++) {
        cuda->SetDmat(d, r, dMat(d, r));
      }
    }
  }
  try {
    bool gwr_status = cuda->Regression(hatmatrix, p, theta, longlat, bw, kernel, adaptive, ngroup, gpuID);
    if (gwr_status) {
      if (hatmatrix) {
        mat betas(N, K, fill::zeros);
        mat betasSE(N, K, fill::zeros);
        vec s_hat(2, fill::zeros);
        vec qdiag(N, fill::zeros);
        for (int r = 0; r < N; r++) {
          for (int c = 0; c < K; c++) {
            betas(r, c) = cuda->GetBetas(r, c);
            betasSE(r, c) = cuda->GetBetasSE(r, c);
          }
          qdiag(r) = cuda->GetQdiag(r);
        }
        s_hat(0) = cuda->GetShat1();
        s_hat(1) = cuda->GetShat2();
        __result = Rcpp::wrap(Rcpp::List::create(
          Named("betas") = betas,
          Named("betas.SE") = betasSE,
          Named("s_hat") = s_hat,
          Named("q.diag") = qdiag
        ));
      } else {
        mat betas(n, K, fill::zeros);
        for (int r = 0; r < n; r++) {
          for (int c = 0; c < K; c++) {
            betas(r, c) = cuda->GetBetas(r, c);
          }
        }
        __result = Rcpp::wrap(Rcpp::List::create(
          Named("betas") = betas
        ));
      }
    } else {
      __result = Rcpp::wrap(false);
    }
    GWCUDA_Del(cuda);
  } catch (std::exception &ex) {
    throw ex;
  }
  return __result;
  END_RCPP
}
#else
RcppExport SEXP GWmodel_gw_reg_cuda(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP rp_givenSEXP, SEXP rpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, SEXP hatmatrixSEXP, 
                                   SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                   SEXP ngroupSEXP, SEXP gpuIDSEXP) {
  BEGIN_RCPP
  throw exception("Method NOT implemented");
  END_RCPP
}
#endif

//GWmodel_gw_reg_all_omp
#ifdef _OPENMP
Rcpp::List gw_reg_all_omp(mat x, vec y, mat dp, bool rp_given, mat rp, bool dm_given, mat dmat, bool hatmatrix, 
                          double p, double theta, bool longlat, 
                          double bw, int kernel, bool adaptive,
                          int threads, int ngroup, int igroup);
RcppExport SEXP GWmodel_gw_reg_all_omp(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP rp_givenSEXP, SEXP rpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, SEXP hatmatrixSEXP, 
                                       SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                       SEXP threadsSEXP, SEXP ngroupSEXP, SEXP igroupSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dp(dpSEXP);
  Rcpp::traits::input_parameter< bool >::type rp_given(rp_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type rp(rpSEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dmat(dmatSEXP);
  Rcpp::traits::input_parameter< bool >::type hatmatrix(hatmatrixSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
  Rcpp::traits::input_parameter< int >::type ngroup(ngroupSEXP);
  Rcpp::traits::input_parameter< int >::type igroup(igroupSEXP);
  __result = Rcpp::wrap(gw_reg_all_omp(x, y, dp, rp_given, rp, dm_given, dmat, hatmatrix, p, theta, longlat, bw, kernel, adaptive, threads, ngroup, igroup));
  return __result;
  END_RCPP
}
#else
RcppExport SEXP GWmodel_gw_reg_all_omp(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP rp_givenSEXP, SEXP rpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, SEXP hatmatrixSEXP, 
                                       SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                       SEXP threadsSEXP, SEXP ngroupSEXP, SEXP igroupSEXP) {
  BEGIN_RCPP
  throw exception("Method NOT implemented");
  END_RCPP
}
#endif

//GWmodel_gw_cv_all
double gw_cv_all(mat x, vec y, mat dp, bool dm_given, mat dmat, 
                 double p, double theta, bool longlat, 
                 double bw, int kernel, bool adaptive,
                 int ngroup, int igroup);
RcppExport SEXP GWmodel_gw_cv_all(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, 
                                  SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                  SEXP ngroupSEXP, SEXP igroupSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dp(dpSEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dmat(dmatSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  Rcpp::traits::input_parameter< int >::type ngroup(ngroupSEXP);
  Rcpp::traits::input_parameter< int >::type igroup(igroupSEXP);
  try {
    __result = Rcpp::wrap(gw_cv_all(x, y, dp, dm_given, dmat, p, theta, longlat, bw, kernel, adaptive, ngroup, igroup));
  } catch (std::runtime_error &ex) {
    __result = Rcpp::wrap(R_PosInf);
  } catch (std::exception &ex) {
    throw ex;
  }
  return __result;
  END_RCPP
}

//GWmodel_gw_cv_all
#ifdef _OPENMP
double gw_cv_all_omp(mat x, vec y, mat dp, bool dm_given, mat dmat, 
                     double p, double theta, bool longlat, 
                     double bw, int kernel, bool adaptive,
                     int threads, int ngroup, int igroup);
RcppExport SEXP GWmodel_gw_cv_all_omp(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, 
                                      SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                      SEXP threadsSEXP, SEXP ngroupSEXP, SEXP igroupSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dp(dpSEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dmat(dmatSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
  Rcpp::traits::input_parameter< int >::type ngroup(ngroupSEXP);
  Rcpp::traits::input_parameter< int >::type igroup(igroupSEXP);
  try {
    __result = Rcpp::wrap(gw_cv_all_omp(x, y, dp, dm_given, dmat, p, theta, longlat, bw, kernel, adaptive, threads, ngroup, igroup));
  } catch (std::runtime_error &ex) {
    __result = Rcpp::wrap(R_PosInf);
  } catch (std::exception &ex) {
    throw ex;
  }
  return __result;
  END_RCPP
}
#else
RcppExport SEXP GWmodel_gw_cv_all_omp(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, 
                                      SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                      SEXP threadsSEXP, SEXP ngroupSEXP, SEXP igroupSEXP) {
  BEGIN_RCPP
  throw exception("Method NOT implemented");
  END_RCPP
}
#endif

#ifdef CUDA_ACCE
RcppExport SEXP GWmodel_gw_cv_all_cuda(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, 
                                   SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                   SEXP ngroupSEXP, SEXP gpuIDSEXP) {
  BEGIN_RCPP
  Rcpp::RObject __result;
  Rcpp::RNGScope __rngScope;
  Rcpp::traits::input_parameter< arma::mat >::type xT(xSEXP);
  Rcpp::traits::input_parameter< arma::vec >::type yT(ySEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dpT(dpSEXP);
  Rcpp::traits::input_parameter< bool >::type dm_given(dm_givenSEXP);
  Rcpp::traits::input_parameter< arma::mat >::type dMatT(dmatSEXP);
  Rcpp::traits::input_parameter< double >::type p(pSEXP);
  Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
  Rcpp::traits::input_parameter< bool >::type longlat(longlatSEXP);
  Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
  Rcpp::traits::input_parameter< double >::type kernel(kernelSEXP);
  Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
  Rcpp::traits::input_parameter< int >::type ngroup(ngroupSEXP);
  Rcpp::traits::input_parameter< int >::type gpuID(gpuIDSEXP);
  
  mat x = mat(xT);
  mat y = mat(yT);
  mat dp = mat(dpT);
  mat dMat = mat(dMatT);
  int N = x.n_rows;
  int K = x.n_cols;
  int n = dp.n_rows;
  try {
    IGWmodelCUDA* cuda = GWCUDA_Create(N, K, false, n, dm_given);
    for (int r = 0; r < N; r++) {
      for (int c = 0; c < K; c++) {
        cuda->SetX(c, r, x(r, c));
      }
      cuda->SetY(r, y(r));
      cuda->SetDp(r, dp(r, 0), dp(r, 1));
    }
    if (dm_given) {
      for (int d = 0; d < N; d++) {
        for (int r = 0; r < n; r++) {
          cuda->SetDmat(d, r, dMat(d, r));
        }
      }
    }
    double cv = cuda->CV(p, theta, longlat, bw, kernel, adaptive, ngroup, gpuID);
    if (cv < DBL_MAX) {
      __result = Rcpp::wrap(cv);
    } else {
      __result = Rcpp::wrap(R_PosInf);
    }
    GWCUDA_Del(cuda);
  } catch (std::exception &ex) {
    throw ex;
  }
  return __result;
  END_RCPP
}
#else
RcppExport SEXP GWmodel_gw_cv_all_cuda(SEXP xSEXP, SEXP ySEXP, SEXP dpSEXP, SEXP dm_givenSEXP, SEXP dmatSEXP, 
                                   SEXP pSEXP, SEXP thetaSEXP, SEXP longlatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP,
                                   SEXP ngroupSEXP, SEXP gpuIDSEXP) {
  BEGIN_RCPP
  throw exception("Method NOT implemented");
  END_RCPP
}
#endif

// e_vec
arma::vec e_vec(int m, int n);
RcppExport SEXP GWmodel_e_vec(SEXP mSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    rcpp_result_gen = Rcpp::wrap(e_vec(m, n));
    return rcpp_result_gen;
END_RCPP
}

// gwr_mixed_trace
double gwr_mixed_trace(arma::mat x1, arma::mat x2, arma::vec y, arma::mat dMat, double bw, std::string kernel, bool adaptive);
RcppExport SEXP GWmodel_gwr_mixed_trace(SEXP x1SEXP, SEXP x2SEXP, SEXP ySEXP, SEXP dMatSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x2(x2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dMat(dMatSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(gwr_mixed_trace(x1, x2, y, dMat, bw, kernel, adaptive));
    return rcpp_result_gen;
END_RCPP
}
// gwr_mixed_2
List gwr_mixed_2(arma::mat x1, arma::mat x2, arma::vec y, arma::mat dMat, arma::mat dMat_rp, double bw, std::string kernel, bool adaptive);
RcppExport SEXP GWmodel_gwr_mixed_2(SEXP x1SEXP, SEXP x2SEXP, SEXP ySEXP, SEXP dMatSEXP, SEXP dMat_rpSEXP, SEXP bwSEXP, SEXP kernelSEXP, SEXP adaptiveSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x1(x1SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type x2(x2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dMat(dMatSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dMat_rp(dMat_rpSEXP);
    Rcpp::traits::input_parameter< double >::type bw(bwSEXP);
    Rcpp::traits::input_parameter< std::string >::type kernel(kernelSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptive(adaptiveSEXP);
    rcpp_result_gen = Rcpp::wrap(gwr_mixed_2(x1, x2, y, dMat, dMat_rp, bw, kernel, adaptive));
    return rcpp_result_gen;
END_RCPP
}
