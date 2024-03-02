#include "utils.hpp"

double* Utils::P2C_mat(Matrix<double> a) {
    std::vector<size_t> shape = a.shape();
	size_t n = shape[0]; //row
	size_t m = shape[1]; //col
	double* a_c = (double*)malloc(sizeof(double)*n*m);

	for(size_t i = 0; i < n; ++i) {
		for(size_t j = 0; j < m; ++j) {
        		a_c[i*m + j] = a(i,j);
		}
  	}
	return a_c;
}


double* Utils::P2C_vec(std::vector<double> a) {
	size_t n = a.size();
	double* a_c = (double*)malloc(sizeof(double)*n);
	for(size_t i = 0; i < n; ++i) {
        	a_c[i] = a[i];
  	}
	return a_c;
}