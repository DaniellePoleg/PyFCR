#pragma once

#include "pybind_matrix.hpp"

namespace Utils {
    double* P2C_mat(Matrix<double> a);
    double* P2C_vec(std::vector<double> a);
}
