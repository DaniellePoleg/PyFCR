import random

from .fcr_base import *
from .utilities import *
from ._estimators import *


class BiometricsDataModel(FCRDataModel):
    # todo: add a function to validate data
    def get_n_covariates_from_beta(self):
        return 1

    def get_z_dimension(self):
        return (self.n_clusters, self.n_members, self.n_covariates)

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
            censoring_time = self.censoring_method(1)
            time = min(T_0, censoring_time)
            if time == censoring_time:  # if censored
                event_type = 0
                delta = [0] * self.n_competing_risks
        else:
            time = T_0
        return event_type, delta, time, covariates

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
        self.beta_coefficients_estimators = np.zeros((self.model.n_covariates, self.model.n_competing_risks, 1),
                                                     dtype=float)

    def get_delta_sums(self):
        return np.vstack([self.model.deltas[:, :, i] for i in range(self.model.n_competing_risks)])

    def calculate_frailty_covariance_estimators(self, args):
        return calculate_frailty_covariance_estimators_biometrics_c(*args)

    def calculate_frailty_exponent_estimators(self, args):
        return calculate_frailty_exponent_estimators_biometrics_c(*args)

    def get_beta_z(self):
        beta_z = np.empty(shape=(self.model.n_clusters, self.model.n_members, self.model.n_competing_risks),
                          dtype=float)
        for j in range(self.model.n_members):
            for i in range(self.model.n_competing_risks):
                beta_z[:, j, i] = np.dot(self.beta_coefficients_estimators[i], self.model.Z[:, j, :].T)
        return np.concatenate([beta_z[:, :, i] for i in range(self.model.n_competing_risks)])

    def get_cox_estimators(self):
        beta_coefficients = np.empty(shape=(self.model.n_competing_risks, self.model.n_covariates), dtype=float)
        hazard_at_event = np.empty(shape=(self.model.n_clusters, self.model.n_members, self.model.n_competing_risks),
                                   dtype=float)
        cumulative_hazards = []

        frailty_exponent = np.repeat(self.frailty_exponent, repeats=self.model.n_members, axis=0)
        cox_weights = np.repeat(self.random_cox_weights, self.model.n_members)
        X = self.model.X.reshape(-1)

        for j in range(self.model.n_competing_risks):
            cur_delta = self.model.deltas[:, :, j].reshape(-1)
            formula, data = self.get_survival_formula_and_data(X, cur_delta, frailty_exponent, j)
            cur_beta_coefficients, hazard, times = parse_cox_estimators(formula, data, cox_weights)
            beta_coefficients[j, :] = cur_beta_coefficients
            cumulative_hazard = self.get_cumulative_hazards(hazard, cur_beta_coefficients)
            cumulative_hazards.append(pd.DataFrame({'x': times, 'y': cumulative_hazard}))
            hazard_at_event[:, :, j] = get_hazard_at_event(X, times, cumulative_hazard).reshape((self.model.n_clusters,
                                                                                                 self.model.n_members))
        hazard_at_event = np.concatenate([hazard_at_event[:, :, i] for i in range(self.model.n_competing_risks)])
        return beta_coefficients, hazard_at_event, cumulative_hazards

    def get_z_mean(self):
        return self.model.Z.reshape(-1, self.model.n_covariates).T.mean(axis=1)

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

    def print_betas(self, mean, standard_error, standard_deviation) -> None:
        def getter(param, i):
            return param[i, :].reshape(-1)
        df = pd.DataFrame(columns=[f'beta_covariate{i}' for i in range(1, self.model.n_covariates + 1)])
        self.print_estimators(df, mean, standard_error, standard_deviation, getter)

    def print_cumulative_hazards(self, mean, standard_error, standard_deviation) -> None:
        def getter(param, i):
            return param[i, :].reshape(-1)
        df = pd.DataFrame(
            columns=[f'cumulative_hazard_threshold_{i}' for i in range(1, self.model.n_threshold_cum_hazard + 1)])
        self.print_estimators(df, mean, standard_error, standard_deviation, getter)


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
        self.analyze_statistical_results(empirical_run=True, multi_estimators_df=self.multi_estimators_df)
