#pragma once

#include "pybind_matrix.hpp"

namespace Assaf {
    void calculate_frailty_covariance_estimators_assaf(
            double *rowSums,
            double *points,
            double *weights,
            double *hazared_at_event,
            double *beta_Z,
            int N,
            int N_Q,
            int N_E,
            int N_P,
            double *W,
            double *ret);

    Matrix<double> calculate_frailty_covariance_estimators_assaf_C(
            const Matrix<double> &rowSums,
            const Matrix<double> &P,
            const std::vector<double> &weights,
            const Matrix<double> &hazared_at_event,
            const Matrix<double> &beta_Z,
            const std::vector<double> &W);

    void calculate_frailty_exponent_estimators_assaf(
            double *rowSums,
            double *points,
            double *weights,
            double *hazared_at_event,
            double *beta_Z,
            int N,
            int N_Q,
            int N_E,
            int N_P,
            double *ret);

    Matrix<double> calculate_frailty_exponent_estimators_assaf_C(
            const Matrix<double> &rowSums,
            const Matrix<double> &P,
            const std::vector<double> &weights,
            const Matrix<double> &hazared_at_event,
            const Matrix<double> &beta_Z);
}