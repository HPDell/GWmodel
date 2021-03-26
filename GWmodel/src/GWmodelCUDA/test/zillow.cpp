#include "zillow.h"

bool read_data(mat& coords, mat& data, vector<string>& fields)
{
    field<std::string> headers = { "utmX", "utmY", "value", "nbaths", "nbeds", "area", "age"};
    mat zillow;
    bool loaded = zillow.load(arma::csv_name(string(SAMPLE_DATA_DIR) + "/zillow.csv", headers));

    if (loaded)
    {
        coords = zillow.cols(0, 1);
        data = zillow.cols(2, headers.n_elem - 1);
        fields = { "value", "nbaths", "nbeds", "area", "age" };
        return true;
    }
    else return false;
}