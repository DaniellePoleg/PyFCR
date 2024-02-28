import math
import random
import os

from .fcr_base import *
from .utilities import *
from ._estimators import *
from .config import Config


class BiometricsDataModel(FCRDataModel):
    # todo: add a function to validate data
    def __init__(self, config: Config):
        super().__init__(config)
        if self.run_type == RunMode.SIMULATION:
            if len(self.beta_coefficients.shape) > 1:
                self.n_covariates = self.beta_coefficients.shape[1]
                assert self.beta_coefficients.shape[0] == self.n_competing_risks
            else:
                assert self.beta_coefficients.shape[0] == self.n_competing_risks
            self.deltas = np.empty(shape=(self.n_clusters, self.n_members, self.n_competing_risks), dtype=bool)
            self.event_types = np.empty(shape=(self.n_clusters, self.n_members), dtype=int)
            self.X = np.empty(shape=(self.n_clusters, self.n_members), dtype=float)
            self.Z = np.empty((self.n_clusters, self.n_members, self.n_covariates), dtype=float)

            assert len(self.frailty_mean) == self.n_competing_risks
            assert len(self.frailty_covariance) == self.n_competing_risks

    def read_data(self, data_path):
        self.deltas = np.loadtxt(os.path.join(data_path, "deltas.csv"), delimiter=',').reshape((self.n_clusters, self.n_members, self.n_competing_risks))
        self.Z = np.loadtxt(os.path.join(data_path, "Z.csv"), delimiter=',').reshape((self.n_clusters, self.n_members, self.n_covariates))
        self.X = np.loadtxt(os.path.join(data_path, "X.csv"), delimiter=',').reshape((self.n_clusters, self.n_members))

    def simulate_data(self):
        random.seed(10)
        for k in range(self.n_clusters):
            frailty_variates = np.random.multivariate_normal(self.frailty_mean, self.frailty_covariance)
            for j in range(self.n_members):
                event_type, delta, time, covariates = self.simulate_data_single_cluster(frailty_variates)
                self.event_types[k][j] = event_type
                self.deltas[k][j] = delta
                self.X[k][j] = time
                self.Z[k, j, :] = covariates.reshape(-1)

    def simulate_data_single_cluster(self, frailty_variates):
        covariates = self.get_covariates_single_simulation()
        gammas = self.get_gammas_single_simulation(covariates, frailty_variates)
        gamma = gammas.sum()
        T_0 = np.random.exponential(1 / gamma, 1)
        delta = np.random.multinomial(n=1, pvals=[gammas[i] / gamma for i in range(self.n_competing_risks)], size=1)
        event_type = int(np.where(delta[0] == 1)[0]) + 1
        if self.censoring_method:
            censoring_time = self.censoring_method()
            time = min(T_0, censoring_time)
            if time == censoring_time:  # if censored
                event_type = 0
                delta = [0] * self.n_competing_risks
        else:
            time = T_0
        return event_type, delta, time, covariates

    def get_covariates_single_simulation(self):
        covariates = np.empty(shape=(self.n_covariates, 1), dtype=float)
        if self.uniform:
            for i in range(self.n_covariates):
                covariates[i] = np.random.uniform(0, 1)
        else:
            for i in range(self.n_covariates):
                covariates[i] = np.random.randint(0, 2)
        return covariates

    def get_gammas_single_simulation(self, covariates, frailty_variates):
        gammas = np.empty(shape=self.n_competing_risks, dtype=float)
        for i in range(self.n_competing_risks):
            if self.n_covariates == 1:
                beta_z = np.dot(self.beta_coefficients[i], covariates)
            else:
                beta_z = np.dot(self.beta_coefficients[i, :].reshape(-1), covariates)
            gammas[i] = np.exp(beta_z + frailty_variates[i])
        return gammas


class BiometricsRunner(Runner):
    def __init__(self, dataModel):
        super().__init__(dataModel)
        self.model.__class__ = BiometricsDataModel
        self.frailty_exponent = np.ones((self.model.n_clusters, self.model.n_competing_risks), dtype=float)
        self.beta_coefficients_estimators = np.zeros((self.model.n_covariates, self.model.n_competing_risks, 1), dtype=float)
        self.frailty_covariance_estimators = np.diag(np.ones(self.model.n_competing_risks))
        self.cumulative_hazards_estimators = []
        self.estimators_df = pd.DataFrame(columns=['betas', 'frailty_covariance', 'cumulative_hazards'])

    def bootstrap_run(self):
        for boot in range(self.model.n_bootstrap):
            random_cox_weights = np.random.exponential(size=(self.model.n_clusters,))
            self.random_cox_weights = random_cox_weights / random_cox_weights.mean()
            self.single_run()
            self.estimators_df.loc[boot] = [self.beta_coefficients_estimators, self.frailty_covariance_estimators,
                                       self.get_cumulative_hazard_estimators()
                                       ]
        self.beta_coefficients_res, self.frailty_covariance_res, self.cumulative_hazards_res = self.reshape_estimators_from_df(
            self.estimators_df, self.model.n_bootstrap)

    def single_run(self):
        iteration_cnt = 0
        convergence = 1
        while (convergence > self.convergence_threshold) & (iteration_cnt < self.max_iterations):
            old_betas = self.beta_coefficients_estimators
            old_frailty_covariance = self.frailty_covariance_estimators
            try:
                self.beta_coefficients_estimators, hazard_at_event, self.cumulative_hazards_estimators = \
                    self.get_cox_estimators(self.frailty_exponent, cox_weights=self.random_cox_weights)
            except Exception as e:
                print("Failed in get_cox_estimators in run_single_estimation, The error is: ", e)
            self.frailty_covariance_estimators = self.get_frailty_covariance_estimators(hazard_at_event)
            self.frailty_exponent = self.get_frailty_exponent_estimators(hazard_at_event)
            convergence = get_estimators_convergence(old_betas, self.beta_coefficients_estimators,
                                                     old_frailty_covariance, self.frailty_covariance_estimators)
            if math.isnan(convergence):
                break
            iteration_cnt += 1

    def get_estimators_arguments(self):
        rowSums = np.vstack([self.model.deltas[:, :, i] for i in range(self.model.n_competing_risks)])
        gauss_hermite_points, gauss_hermite_weights = multi_gauss_hermite_calculation(
            sigma=self.frailty_covariance_estimators, pts=self.points_for_gauss_hermite,
            wts=self.weights_for_gauss_hermite)
        beta_z = self.get_beta_z(self.beta_coefficients_estimators)
        return rowSums, gauss_hermite_points, gauss_hermite_weights, beta_z

    def get_frailty_covariance_estimators(self, hazard_at_event):
        rowSums, gauss_hermite_points, gauss_hermite_weights, beta_z = self.get_estimators_arguments()
        return calculate_frailty_covariance_estimators_biometrics_c(rowSums, gauss_hermite_points, gauss_hermite_weights, hazard_at_event, beta_z, self.random_cox_weights)

    def get_frailty_exponent_estimators(self, hazard_at_event):
        rowSums, gauss_hermite_points, gauss_hermite_weights, beta_z = self.get_estimators_arguments()
        return calculate_frailty_exponent_estimators_biometrics_c(rowSums, gauss_hermite_points,
                                                                     gauss_hermite_weights, hazard_at_event, beta_z)

    def get_beta_z(self, beta_hat):
        beta_z = np.empty(shape=(self.model.n_clusters, self.model.n_members, self.model.n_competing_risks),
                          dtype=float)
        for j in range(self.model.n_members):
            for i in range(self.model.n_competing_risks):
                beta_z[:, j, i] = np.dot(beta_hat[i], self.model.Z[:, j, :].T)
        return np.concatenate([beta_z[:, :, i] for i in range(self.model.n_competing_risks)])

    def get_cox_estimators(self, frailty_exponent, cox_weights):
        beta_coefficients = np.empty(shape=(self.model.n_competing_risks, self.model.n_covariates), dtype=float)
        hazard_at_event = np.empty(shape=(self.model.n_clusters, self.model.n_members, self.model.n_competing_risks),
                                   dtype=float)
        cumulative_hazards = []

        frailty_exponent = np.repeat(frailty_exponent, repeats=self.model.n_members, axis=0)
        cox_weights = np.repeat(cox_weights, self.model.n_members)
        X = self.model.X.reshape(-1)

        for j in range(self.model.n_competing_risks):
            formula, data = self.get_survival_formula_and_data(X, frailty_exponent, j)
            cur_beta_coefficients, hazard, times = parse_cox_estimators(formula, data, cox_weights)
            beta_coefficients[j, :] = cur_beta_coefficients
            cumulative_hazard = self.get_cumulative_hazards(hazard, cur_beta_coefficients)
            cumulative_hazards.append(pd.DataFrame({'x': times, 'y': cumulative_hazard}))
            hazard_at_event[:, :, j] = get_hazard_at_event(X, times, cumulative_hazard).reshape((self.model.n_clusters,
                                                                                                 self.model.n_members))
        hazard_at_event = np.concatenate([hazard_at_event[:, :, i] for i in range(self.model.n_competing_risks)])
        return beta_coefficients, hazard_at_event, cumulative_hazards

    def get_cumulative_hazards(self, hazard, beta_coefficients):
        if self.model.uniform:
            z_mean = self.model.Z.reshape(-1, self.model.n_covariates).T.mean(axis=1)
            cumulative_hazard = list(np.cumsum(hazard / np.exp(np.dot(beta_coefficients, z_mean))))
        else:
            cumulative_hazard = list(np.cumsum(hazard))
        return cumulative_hazard

    def get_cumulative_hazard_estimators(self):
        cumulative_hazards_at_points = np.empty(
            shape=(len(self.cumulative_hazards_estimators), len(self.model.cumulative_hazard_thresholds)), dtype=float)
        for i, cumulative_hazard in enumerate(self.cumulative_hazards_estimators):
            for j, threshold in enumerate(self.model.cumulative_hazard_thresholds):
                if not cumulative_hazard[cumulative_hazard.x > threshold].empty:
                    cumulative_hazards_at_points[i, j] = cumulative_hazard.y[
                        cumulative_hazard[cumulative_hazard.x > threshold].index[0]]
        return cumulative_hazards_at_points

    def get_estimators_dimensions(self, n_repeats):
        return [(n_repeats, self.model.n_competing_risks, self.model.n_covariates),
                  (n_repeats, self.model.n_competing_risks, self.model.n_competing_risks),
                  (n_repeats, self.model.n_competing_risks, self.model.n_threshold_cum_hazard)]

    def analyze_statistical_results(self, empirical_run):
        axis = 0
        mean_beta_coefficients = self.beta_coefficients_res.mean(axis=axis)
        mean_frailty_covariance = self.frailty_covariance_res.mean(axis=axis)
        mean_cumulative_hazards = np.nanmean(self.cumulative_hazards_res, axis=axis)

        if empirical_run:
            var_beta_coefficients = self.calculate_empirical_variance(self.beta_coefficients_res,
                                                                 mean_beta_coefficients)
            var_frailty_covariance = self.calculate_empirical_variance(self.frailty_covariance_res,
                                                                  mean_frailty_covariance)
            var_cumulative_hazard = self.calculate_empirical_variance(self.cumulative_hazards_res,
                                                                 mean_cumulative_hazards)
            columns = [[f'j{i}_param', f'j{i}_SE'] for i in range(1, self.model.n_competing_risks+1)]
            columns = [item for sublist in columns for item in sublist]

            df = pd.DataFrame(columns=columns)
            # df.loc['betas'] = [[mean_beta_coefficients[i][0], var_beta_coefficients[i][0]] for i in range(self.model.n_competing_risks)]

            df.loc['betas'] = [mean_beta_coefficients[0][0], var_beta_coefficients[0][0], mean_beta_coefficients[1][0],
                              var_beta_coefficients[1][0]]
            df.loc['frailty_variance'] = [mean_frailty_covariance[0][0], var_frailty_covariance[0][0], mean_frailty_covariance[1][1],
                              var_frailty_covariance[1][1]]
            df.loc['frailty_covariance'] = [mean_frailty_covariance[0][1], var_frailty_covariance[0][1], mean_frailty_covariance[1][0],
                              var_frailty_covariance[1][0]]
            df.loc['cumulativa_hazard_0'] = [mean_cumulative_hazards[0][0], var_cumulative_hazard[0][0], mean_cumulative_hazards[1][0],
                              var_cumulative_hazard[1][0]]
            df.loc['cumulativa_hazard_1'] = [mean_cumulative_hazards[0][1], var_cumulative_hazard[0][1], mean_cumulative_hazards[1][1],
                              var_cumulative_hazard[1][1]]
            df.loc['cumulativa_hazard_2'] = [mean_cumulative_hazards[0][2], var_cumulative_hazard[0][2], mean_cumulative_hazards[1][2],
                              var_cumulative_hazard[1][2]]
            df.loc['cumulativa_hazard_3'] = [mean_cumulative_hazards[0][3], var_cumulative_hazard[0][3], mean_cumulative_hazards[1][3],
                              var_cumulative_hazard[1][3]]
            print(df.round(4))
            # print("mean_beta_coefficients: ", np.round(mean_beta_coefficients, 4))
            # print("emp_var_beta_coefficients: ", np.round(var_beta_coefficients, 4))
            # print("mean_frailty_covariance: ", np.round(mean_frailty_covariance, 4))
            # print("emp_var_frailty_covariance: ", np.round(var_frailty_covariance, 4))
            # print("mean_cumulative_hazards: ", np.round(mean_cumulative_hazards, 4))
            # print("emp_var_cumulative_hazard: ", np.round(var_cumulative_hazard, 4))

        else:
            var_beta_coefficients = self.beta_coefficients_res.var(axis=axis)
            var_frailty_covariance = self.frailty_covariance_res.var(axis=axis)
            var_cumulative_hazard = np.nanvar(self.cumulative_hazards_res, axis=axis)

        return mean_beta_coefficients, mean_frailty_covariance, mean_cumulative_hazards, var_beta_coefficients, \
            var_frailty_covariance, var_cumulative_hazard


class BiometricsMultiRunner(BiometricsRunner):
    def __init__(self, model):
        super().__init__(model)
        self.multi_estimators_df = pd.DataFrame(columns=['betas', 'sigmas', 'cums_res', 'betas_vars', 'sigmas_vars', 'cums_vars'])

    def run(self):
        cnt_columns_coverage_rates = 2 * (self.model.n_covariates*self.model.n_competing_risks + self.model.n_competing_risks * self.model.n_competing_risks +
                                          self.model.n_competing_risks * self.model.n_threshold_cum_hazard)
        coverage_rates_df = pd.DataFrame(columns=range(cnt_columns_coverage_rates))
        event_types_analysis = []
        for i in range(self.model.n_simulations):
            print("simulation number: ", i)
            self.model.simulate_data()
            self.bootstrap_run()
            # get analysis of bootstrap results
            self.multi_estimators_df.loc[i] = [*self.analyze_statistical_results(empirical_run=False)]
            coverage_rates_df.loc[i] = self.get_multiple_confidence_intervals()
            if self.model.n_competing_risks == 2:
                event_types_analysis = self.calculate_event_types_results(event_types_analysis)
            # todo: add some print for every simulation - it takes too long the whole simulation with not enough indications
        self.beta_coefficients_res, self.frailty_covariance_res, self.cumulative_hazards_res = self.reshape_estimators_from_df(
            self.multi_estimators_df, self.model.n_simulations)

    #todo: this can be static
    def calculate_event_types_results(self, all_results):
        censoring_rate_1 = sum([int(i[0] == 0) for i in self.model.event_types])
        censoring_rate_2 = sum([int(i[1] == 0) for i in self.model.event_types])
        censoring = np.array(censoring_rate_1) + np.array(censoring_rate_2)
        event_type_1 = sum([int(i[0] == 1) + int(i[1] == 1) for i in self.model.event_types])
        event_type_2 = sum([int(i[0] == 2) + int(i[1] == 2) for i in self.model.event_types])
        event_type_different = sum([(int(i[0] == 1 and i[1] == 2) or int(i[0] == 2 and i[1] == 1)) for i in self.model.event_types])
        event_type_both_1 = sum([int(i[0] == 1 and i[1] == 1) for i in self.model.event_types])
        event_type_both_2 = sum([int(i[0] == 2 and i[1] == 2) for i in self.model.event_types])
        all_results.append((censoring, event_type_1, event_type_2, event_type_different,
                            event_type_both_1, event_type_both_2))
        return all_results

    def print_summary(self) -> None:
        super().print_summary()
        self.calculate_bootstrap_variance()

    def calculate_bootstrap_variance(self) -> None:
        betas_vars, sigmas_vars, cums_vars = self.reshape_estimators_from_df(self.multi_estimators_df.iloc[:, 3:6],
                                                                                    self.model.n_simulations)
        mean_betas_vars = np.mean(betas_vars, axis=0)
        mean_sigmas_vars = np.mean(sigmas_vars, axis=0)
        mean_cums_vars = np.nanmean(cums_vars, axis=0)

        df = pd.DataFrame(columns=['j1_param', 'j1_SE', 'j2_param', 'j2_SE'])
        df.loc['betas_vars'] = [mean_betas_vars[0][0], 'X', mean_betas_vars[1][0], 'X']
        df.loc['sigmas_vars'] = [mean_sigmas_vars[0][0], 'X', mean_sigmas_vars[1][1], 'X']
        df.loc['covariance_vars'] = [mean_sigmas_vars[0][1], 'X', mean_sigmas_vars[1][0], 'X']
        df.loc['cums_vars_0'] = [mean_cums_vars[0][0], 'X', mean_cums_vars[1][0], 'X']
        df.loc['cums_vars_1'] = [mean_cums_vars[0][1], 'X', mean_cums_vars[1][1], 'X']
        df.loc['cums_vars_2'] = [mean_cums_vars[0][2], 'X', mean_cums_vars[1][2], 'X']
        df.loc['cums_vars_3'] = [mean_cums_vars[0][3], 'X', mean_cums_vars[1][3], 'X']
        print(df.round(4))

        # print("variance of betas from bootstrap: ", np.round(np.mean(betas_vars, axis=0), 4))
        # print("variance of sigmas from bootstrap: ", np.round(np.mean(sigmas_vars, axis=0), 4))
        # print("variance of cumulative hazards from bootstrap: ", np.round(np.nanmean(cums_vars, axis=0), 4))

        # print("results for run of n clusters:", self.model.n_clusters)
        # if event_types:
        #     print("average:", [np.round(np.mean(value), 2) for value in event_types])
        #     print("SD:", [np.round(np.std(value), 2) for value in event_types])

        # todo: add try and catch if results dir does not exist
        # coverage_rates_df.to_csv("results\\conf_intervals.csv")

        # todo: parse dataframe to txt
        # np.savetxt("results\\beta_coefficient_estimators.csv", betas.reshape(-1), delimiter=",")
        # np.savetxt("results\\frailty_covariance_estimators.csv", sigmas.reshape(-1), delimiter=",")
        # np.savetxt("results\\cumulative_hazard_estimators.csv", cums_res.reshape(-1), delimiter=",")
        # np.savetxt("results\\beta_coefficient_var_boot.csv", betas_vars.reshape(-1), delimiter=",")
        # np.savetxt("results\\frailty_covariance_var_boot.csv", sigmas_vars.reshape(-1), delimiter=",")
        # np.savetxt("results\\cumulative_hazard_var_boot.csv", cums_vars.reshape(-1), delimiter=",")
        # np.loadtxt("results\\deltas.csv", delimiter=',').reshape(500, 2, 2)
        # np.loadtxt("results\\Z.csv", delimiter=',').reshape(500, 2, 1)
        # np.loadtxt("results\\Z.csv", delimiter=',').reshape(500, 2)
        # np.savetxt("data\\deltas.csv", self.model.deltas.reshape(-1), delimiter=",")
