#include <vector>
#include <string>
#include <armadillo>
#include "CGWmodelCUDA.h"
#include "zillow.h"

int main() {
    mat coord, data;
    vector<string> fields;
    read_data(coord, data, fields);
    
    if (coord.n_rows < 1 || data.n_rows < 1 || coord.n_rows != data.n_rows || data.n_cols != fields.size()) {
        std::cerr << "Error read data." << std::endl;
        return 1;
    }

    int N = data.n_rows, K = data.n_cols, n = N;
    mat y = data.col(0), x = data.each_row([](rowvec& r) { r(0) = 1; });
    mat dp = coord, rp = coord;
    bool hatmatrix = true, rp_given = false, dm_given = false, ftest = false, longlat = false, adaptive = false;
    double p = 2.0, theta = 0.0, bw = 10000;
    int kernel = 0, ngroup = 64, gpuID = 0;

    CGWmodelCUDA* cuda = new CGWmodelCUDA(N, K, rp_given, n, dm_given);
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < K; c++) {
            cuda->SetX(c, r, x(r, c));
        }
        cuda->SetY(r, y(r));
        cuda->SetDp(r, dp(r, 0), dp(r, 1));
    } 
    std::cout << "start" << endl;

    try {
        bool gwr_status = cuda->Regression(hatmatrix, ftest, p, theta, longlat, bw, kernel, adaptive, ngroup, gpuID);
        if (gwr_status) {
            std::cout << "get data" << endl;
            mat betas(N, K, fill::zeros);
            mat betasSE(N, K, fill::zeros);
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < K; c++) {
                    betas(r, c) = cuda->GetBetas(r, c);
                    betasSE(r, c) = cuda->GetBetasSE(r, c);
                }
            }
            vec q = {0.0, 0.25, 0.5, 0.75, 1.0};
            mat p = quantile(betas, q, 0);
            p.print("betas");
            vec s_hat = { cuda->GetShat1(), cuda->GetShat2() };
            s_hat.print("s_hat");
        } else {
            std::cerr << "Cuda error:" << gwr_status << std::endl;
        }
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    delete cuda;
}