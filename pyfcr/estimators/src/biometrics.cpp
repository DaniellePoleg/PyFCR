#include <cstring>
#include <cmath>
#include "biometrics.hpp"
#include "utils.hpp"


Matrix<double> Biometrics::calculate_frailty_exponent_estimators_biometrics_C(
        const Matrix<double> &rowSums,
        const Matrix<double> &P,
        const std::vector<double> &weights,
        const Matrix<double> &hazared_at_event,
        const Matrix<double> &beta_Z) {
	size_t N_P = P.shape()[0];
    size_t N_E = P.shape()[1];
	size_t N_Q = hazared_at_event.shape()[1];	size_t N = rowSums.shape()[0]/N_E;	double* rowSums_c = Utils::P2C_mat(rowSums);
	double* P_c = Utils::P2C_mat(P);
	double* W_c = Utils::P2C_vec(weights);
	double* hazared_at_event_c = Utils::P2C_mat(hazared_at_event);
	double* beta_Z_c = Utils::P2C_mat(beta_Z);

	double* ret = (double*)malloc(sizeof(double)*N * N_E);
	memset(ret,0,sizeof(double)*N * N_E);

	calculate_frailty_exponent_estimators_biometrics(rowSums_c,P_c,W_c,hazared_at_event_c,beta_Z_c,N,N_Q,N_E,N_P,ret);

	std::vector<size_t> ndim_vec(2);
    ndim_vec[0] = N;
    ndim_vec[1] = N_E;

    Matrix<double> out(ndim_vec);

	for(int i = 0; i <N; ++i) {
		for(int j = 0; j  < N_E; ++j) {
	    		out(i,j) = ret[i*N_E + j];
		}
	}

	free(rowSums_c);
	free(P_c);
	free(W_c);
	free(ret);
	free(hazared_at_event_c);
	free(beta_Z_c);
	return out;
}

void Biometrics::calculate_frailty_exponent_estimators_biometrics(
        double* rowSums,
        double* points,
        double* weights,
        double* hazared_at_event,
        double* beta_Z,
        int N,
        int N_Q,
        int N_E,
        int N_P,
        double* ret)
{

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N_E; j++)
			ret[i*N_E + j] = 0;
	double* devisor = new double[N];

	for (int i = 0; i < N; i++)
		devisor[i] = 0;
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {		for (int k = 0; k < N_P; k++) {			double integrand = 0;
			for (int z = 0; z < N_Q; z++) {                double in_exp = 0;
                for (int j = 0; j < N_E; j++) {                    int ind = j * N * N_Q + i * N_Q + z;
					in_exp += (points[k * N_E + j] * rowSums[ind]) - (hazared_at_event[ind]*exp(beta_Z[ind] + points[k*N_E + j]));
				}
				integrand += in_exp;
			}
			integrand = exp(integrand)*weights[k];
			devisor[i] += integrand;
			for (int q = 0; q < N_E; q++) {				ret[i*N_E + q] += exp(points[k*N_E + q])*integrand;
			}
		}
	}
	for (int a = 0; a < N; a++) {		for (int b = 0; b < N_E; b++){			ret[a*N_E + b] = ret[a*N_E + b]/devisor[a];
			}
        }
}


void Biometrics::calculate_frailty_covariance_estimators_biometrics(
        double* rowSums,
        double* points,
        double* weights,
        double* hazared_at_event,
        double* beta_Z,
        int N,
        int N_Q,
        int N_E,
        int N_P,
        double* W,
        double* ret)
{
	for (int i = 0; i < N_E; i++){
		for (int j = 0; j < N_E; j++){
			ret[i*N_E + j] = 0;
		}
	}
	double total_W = 0;
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		double devisor = 0;
		double** PU_varcov = new double*[N_E];
        for(int x = 0; x < N_E; ++x) {
            PU_varcov[x] = new double[N_E];
        }
		for (int j = 0; j < N_E; j++) {
			for (int m = 0; m < N_E; m++) {
				PU_varcov[j][m] = 0;
			}
		}
		for (int k = 0; k < N_P; k++) {			double integrand = 0;
            for (int z = 0; z < N_Q; z++) {                double in_exp = 0;
                for (int j = 0; j < N_E; j++) {					int ind = j * N * N_Q + i * N_Q + z;
					in_exp += (points[k * N_E + j] * rowSums[ind]) - (hazared_at_event[ind]*exp(beta_Z[ind] + points[k*N_E + j]));
				}
				integrand += in_exp;
			}
			integrand = exp(integrand)*weights[k];
			devisor += integrand;
			for (int q = 0; q < N_E; q++) {
				for (int p = 0; p < N_E; p++) {
					PU_varcov[q][p] += points[k*N_E + q]*points[k*N_E + p]*integrand;
				}
			}
		}
		for (int a = 0; a < N_E; a++) {
			for (int b = 0; b < N_E; b++) {
				#pragma omp critical
				{
					ret[a*N_E + b] += W[i]*PU_varcov[a][b]/devisor;
				}
			}
		}
		total_W += W[i];
	}
	for (int g = 0; g < N_E; g++) {
		for (int h = 0; h < N_E; h++) {
			{
				ret[g*N_E + h] = ret[g*N_E + h]/total_W;
			}
		}
	}
}

Matrix<double> Biometrics::calculate_frailty_covariance_estimators_biometrics_C(
        const Matrix<double> &rowSums,
        const Matrix<double> &P,
        const std::vector<double> &weights,
        const Matrix<double> &hazared_at_event,
        const Matrix<double> &beta_Z,
        const std::vector<double> &W)
{
	size_t N_P = P.shape()[0];
    size_t N_E = P.shape()[1];
	size_t N_Q = hazared_at_event.shape()[1];	size_t N = rowSums.shape()[0]/N_E;	double* rowSums_c = Utils::P2C_mat(rowSums);
	double* P_c = Utils::P2C_mat(P);
	double* weights_c = Utils::P2C_vec(weights);
	double* hazared_at_event_c = Utils::P2C_mat(hazared_at_event);
	double* beta_Z_c = Utils::P2C_mat(beta_Z);
	double* W_c = Utils::P2C_vec(W);

	double* ret = (double*)malloc(sizeof(double)*N_E * N_E);
	memset(ret,0,sizeof(double)*N_E * N_E);

	calculate_frailty_covariance_estimators_biometrics(rowSums_c,P_c,weights_c,hazared_at_event_c,beta_Z_c,N,N_Q,N_E,N_P,W_c,ret);

    int ndim = 2;
	std::vector<size_t> ndim_vec(ndim);
    for ( size_t i = 0 ; i < ndim ; ++i ){
        ndim_vec[i] = N_E;
    }
    Matrix<double> out(ndim_vec);

	for(int i = 0; i <N_E; ++i) {
		for(int j = 0; j  < N_E; ++j) {
	    		out(i,j) = ret[i*N_E + j];
		}
	}
	free(rowSums_c);
	free(P_c);
	free(weights_c);
	free(ret);
	free(hazared_at_event_c);
	free(beta_Z_c);
	free(W_c);
	return out;
}