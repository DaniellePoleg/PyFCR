import random

from .fcr_base import FCRDataModel, Runner
from .utilities import *
from ._estimators import *

utils = importr('utils')
base = importr('base')
survival = importr('survival')
stats = importr('stats')


class AssafDataModel(FCRDataModel):
    # todo: add a function to validate data
    def get_n_covariates_from_beta(self):
        return 2

    def get_z_dimension(self):
        return (self.n_clusters, 1, self.n_covariates)

    def simulate_data(self):
        random.seed(10)
        for k in range(self.n_clusters):
            frailty_variates = np.random.multivariate_normal(self.frailty_mean, self.frailty_covariance)
            event_type, x, time, covariates = self.simulate_data_single_cluster(frailty_variates)
            self.event_types[k] = event_type
            self.deltas[k] = x
            self.X[k] = time
            self.Z[k, :] = covariates.reshape(-1)

    def simulate_data_single_cluster(self, frailty_variates):
        covariates = self.get_covariates_single_simulation()
        gammas = self.get_gammas_single_simulation(covariates, frailty_variates)
        delta = np.zeros(shape=(self.n_members, self.n_competing_risks))
        T_0 = np.zeros(shape=(self.n_members))
        event_type = np.zeros(self.n_members)
        for i in range(self.n_members):
            gamma = gammas[:, i].sum()
            T_0[i] = np.random.exponential(1 / gamma, 1)
            delta[i, :] = np.random.multinomial(n=1,
                                            pvals=[gammas[j, i] / gamma for j in range(self.n_competing_risks)],
                                            size=1)
            event_type[i] = int(np.where(delta[i] == 1)[0]) + 1

        if self.censoring_method:
            censoring_time = self.censoring_method(self.n_members)
            time = [min(T_0[i], censoring_time[i]) for i in range(self.n_members)]
            for i in range(self.n_members):
                if time[i] == censoring_time[i]:
                    event_type[i] = 0
                    delta[i] = [0] * self.n_competing_risks
        else:
            time = T_0
        return event_type, delta, time, covariates

#todo: check covariates > 1
    def get_gammas_single_simulation(self, covariates, frailty_variates):
        gammas = np.zeros(shape=(self.n_competing_risks, self.n_members))
        for j in range(self.n_competing_risks):
            for i in range(self.n_members):
                # if n_covariates == 1:
                beta_Z = np.dot(self.beta_coefficients[j, i], covariates)  # todo: calc for each member
                # todo: add option for >1 covariates
                # else:
                #     beta_Z = np.dot(beta_coefficients[:, i], covariates)
                gammas[j, i] = np.exp(beta_Z + frailty_variates[j])  # todo: maybe frailty is also for each member i
        return gammas


class AssafRunner(Runner):
    def __init__(self, dataModel):
        super().__init__(dataModel)
        self.model.__class__ = AssafDataModel
        self.beta_coefficients_estimators = np.zeros((self.model.n_covariates, self.model.n_competing_risks, self.model.n_members),
                                                     dtype=float)

    def get_delta_sums(self):
        return self.model.deltas.sum(axis=1)

    def calculate_frailty_covariance_estimators(self, args):
        return calculate_frailty_covariance_estimators_assaf_c(*args)

    def calculate_frailty_exponent_estimators(self, args):
        return calculate_frailty_exponent_estimators_assaf_C(*args)

    def get_cox_estimators(self):
        beta_coefficients_estimators = np.ndarray(
            shape=(self.model.n_competing_risks, self.model.n_members, self.model.n_covariates))
        cumulative_hazards_estimators = []
        hazard_at_event = np.empty(shape=(self.model.n_clusters * self.model.n_competing_risks, self.model.n_members))
        for i in range(self.model.n_members):
            try:
                cur_beta_coefficients, cur_hazard_at_event, cur_cumulative_hazard = self.q_single_observation(i)
            except Exception as e:
                print("Failed in run_multiple_observations in run_single_estimation, The error is: ", e)
                return
            beta_coefficients_estimators[:, i, :] = cur_beta_coefficients
            hazard_at_event[:, i] = cur_hazard_at_event
            cumulative_hazards_estimators.append(cur_cumulative_hazard)
        return beta_coefficients_estimators, hazard_at_event, cumulative_hazards_estimators

    def q_single_observation(self, i):  # runs on single observation i
        beta_coefficients = np.ndarray(shape=(self.model.n_competing_risks, self.model.n_covariates))
        cumulative_hazards = []
        hazard_at_event = np.ndarray(shape=(0,))

        for j in range(self.model.n_competing_risks):  # for j comp risk
            cur_delta = self.model.deltas[:, i, j]
            cur_x = self.model.X[:, i]
            formula, dataframe = self.get_survival_formula_and_data(cur_x, cur_delta, self.frailty_exponent, j)
            cur_beta_coefficients, hazard, times = parse_cox_estimators(formula, dataframe, self.random_cox_weights)
            beta_coefficients[j, :] = cur_beta_coefficients
            cumulative_hazard = self.get_cumulative_hazards(hazard, cur_beta_coefficients)
            cumulative_hazards.append(pd.DataFrame({'x': times, 'y': cumulative_hazard}))
            hazard_at_event = np.concatenate([hazard_at_event, get_hazard_at_event(cur_x, times, cumulative_hazard)])
        return beta_coefficients, hazard_at_event, cumulative_hazards

    def get_z_mean(self):
        return self.model.Z.mean(axis=0).T

    def get_beta_z(self):
        beta_z = np.zeros(shape=(self.model.n_clusters, self.model.n_members, self.model.n_competing_risks))
        for i in range(self.model.n_members):
            for j in range(self.model.n_competing_risks):
                beta_z[:, i, j] = np.dot(self.beta_coefficients_estimators[j, i, :], self.model.Z[:, 0, :].T)
        return np.concatenate([beta_z[:, :, j] for j in range(self.model.n_competing_risks)])

    def get_cumulative_hazard_estimators(self):
        cumulative_hazards_at_points = np.empty(shape=(
        len(self.cumulative_hazards_estimators), len(self.cumulative_hazards_estimators[0]),
        len(self.model.cumulative_hazard_thresholds)),
                                                dtype=float)
        for i, cumulative_hazard in enumerate(self.cumulative_hazards_estimators):  # i members
            for k, current_cumulative_hazard in enumerate(cumulative_hazard):  # k comp_risk
                for j, threshold in enumerate(self.model.cumulative_hazard_thresholds):
                    temp = current_cumulative_hazard[current_cumulative_hazard.x > threshold]
                    if not temp.empty:
                        cumulative_hazards_at_points[i, k, j] = current_cumulative_hazard.y[temp.index[0]]
        return cumulative_hazards_at_points

    def get_estimators_dimensions(self, n_repeats):
        return [(n_repeats, self.model.n_competing_risks, self.model.n_members, self.model.n_covariates),
                (n_repeats, self.model.n_competing_risks, self.model.n_competing_risks),
                (n_repeats, self.model.n_members, self.model.n_competing_risks, self.model.n_threshold_cum_hazard)]

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
            columns = [[f'j{i}_param', f'j{i}_SE'] for i in range(1, self.model.n_competing_risks + 1)]
            columns = [item for sublist in columns for item in sublist]

            df = pd.DataFrame(columns=columns)
            # df.loc['betas'] = [[mean_beta_coefficients[i][0], var_beta_coefficients[i][0]] for i in range(self.model.n_competing_risks)]

            df.loc['betas_m1'] = [mean_beta_coefficients[0][0][0], var_beta_coefficients[0][0][0],
                                  mean_beta_coefficients[1][0][0],
                                  var_beta_coefficients[1][0][0]]
            df.loc['betas_m2'] = [mean_beta_coefficients[0][1][0], var_beta_coefficients[0][1][0],
                                  mean_beta_coefficients[1][1][0],
                                  var_beta_coefficients[1][1][0]]
            df.loc['frailty_variance'] = [mean_frailty_covariance[0][0], var_frailty_covariance[0][0],
                                          mean_frailty_covariance[1][1],
                                          var_frailty_covariance[1][1]]
            df.loc['frailty_covariance'] = [mean_frailty_covariance[0][1], var_frailty_covariance[0][1],
                                            mean_frailty_covariance[1][0],
                                            var_frailty_covariance[1][0]]
            df.loc['cumulative_hazard_0_m1'] = [mean_cumulative_hazards[0][0][0], var_cumulative_hazard[0][0][0],
                                                mean_cumulative_hazards[0][1][0],
                                                var_cumulative_hazard[0][1][0]]
            df.loc['cumulative_hazard_0_m2'] = [mean_cumulative_hazards[1][0][0], var_cumulative_hazard[1][0][0],
                                                mean_cumulative_hazards[1][1][0],
                                                var_cumulative_hazard[1][1][0]]

            df.loc['cumulative_hazard_1_m1'] = [mean_cumulative_hazards[0][0][1], var_cumulative_hazard[0][0][1],
                                                mean_cumulative_hazards[0][1][1],
                                                var_cumulative_hazard[0][1][1]]
            df.loc['cumulative_hazard_1_m2'] = [mean_cumulative_hazards[1][0][1], var_cumulative_hazard[1][0][1],
                                                mean_cumulative_hazards[1][1][1],
                                                var_cumulative_hazard[1][1][1]]

            df.loc['cumulative_hazard_2_m1'] = [mean_cumulative_hazards[0][0][2], var_cumulative_hazard[0][0][2],
                                                mean_cumulative_hazards[0][1][2],
                                                var_cumulative_hazard[0][1][2]]
            df.loc['cumulative_hazard_2_m2'] = [mean_cumulative_hazards[1][0][2], var_cumulative_hazard[1][0][2],
                                                mean_cumulative_hazards[1][1][2],
                                                var_cumulative_hazard[1][1][2]]

            df.loc['cumulative_hazard_3_m1'] = [mean_cumulative_hazards[0][0][3], var_cumulative_hazard[0][0][3],
                                                mean_cumulative_hazards[0][1][3],
                                                var_cumulative_hazard[0][1][3]]
            df.loc['cumulative_hazard_3_m2'] = [mean_cumulative_hazards[1][0][3], var_cumulative_hazard[1][0][3],
                                                mean_cumulative_hazards[1][1][3],
                                                var_cumulative_hazard[1][1][3]]
            print(df.round(4))

        else:
            var_beta_coefficients = self.beta_coefficients_res.var(axis=axis)
            var_frailty_covariance = self.frailty_covariance_res.var(axis=axis)
            var_cumulative_hazard = np.nanvar(self.cumulative_hazards_res, axis=axis)

        return mean_beta_coefficients, mean_frailty_covariance, mean_cumulative_hazards, var_beta_coefficients, \
            var_frailty_covariance, var_cumulative_hazard


class AssafMultiRunner(AssafRunner):
    def __init__(self, model):
        super().__init__(model)
        self.multi_estimators_df = pd.DataFrame(
            columns=['betas', 'sigmas', 'cums_res', 'betas_vars', 'sigmas_vars', 'cums_vars'])

    def run(self):
        cnt_columns_coverage_rates = 2 * (
                    self.model.n_covariates * self.model.n_competing_risks + self.model.n_competing_risks * self.model.n_competing_risks +
                    self.model.n_competing_risks * self.model.n_threshold_cum_hazard)
        # coverage_rates_df = pd.DataFrame(columns=range(cnt_columns_coverage_rates))
        # event_types_analysis = []
        for i in range(self.model.n_simulations):
            print("simulation number: ", i)
            self.model.simulate_data()
            self.bootstrap_run()
            # get analysis of bootstrap results
            self.multi_estimators_df.loc[i] = [*self.analyze_statistical_results(empirical_run=False)]
            # coverage_rates_df.loc[i] = self.get_multiple_confidence_intervals()
            # if self.model.n_competing_risks == 2:
            #     event_types_analysis = self.calculate_event_types_results(event_types_analysis)
            # todo: add some print for every simulation - it takes too long the whole simulation with not enough indications
        self.beta_coefficients_res, self.frailty_covariance_res, self.cumulative_hazards_res = self.reshape_estimators_from_df(
            self.multi_estimators_df, self.model.n_simulations)

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
        df.loc['betas_vars_m1'] = [mean_betas_vars[0][0][0], 'X', mean_betas_vars[1][0][0], 'X']
        df.loc['betas_vars_m2'] = [mean_betas_vars[0][1][0], 'X', mean_betas_vars[1][1][0], 'X']

        df.loc['sigmas_vars'] = [mean_sigmas_vars[0][0], 'X', mean_sigmas_vars[1][1], 'X']
        df.loc['covariance_vars'] = [mean_sigmas_vars[0][1], 'X', mean_sigmas_vars[1][0], 'X']

        df.loc['cums_vars_0_m1'] = [mean_cums_vars[0][0][0], 'X', mean_cums_vars[0][1][0], 'X']
        df.loc['cums_vars_0_m2'] = [mean_cums_vars[1][0][0], 'X', mean_cums_vars[1][1][0], 'X']

        df.loc['cums_vars_1_m1'] = [mean_cums_vars[0][0][1], 'X', mean_cums_vars[0][1][1], 'X']
        df.loc['cums_vars_1_m2'] = [mean_cums_vars[1][0][1], 'X', mean_cums_vars[1][1][1], 'X']

        df.loc['cums_vars_2_m1'] = [mean_cums_vars[0][0][2], 'X', mean_cums_vars[0][1][2], 'X']
        df.loc['cums_vars_2_m2'] = [mean_cums_vars[1][0][2], 'X', mean_cums_vars[1][1][2], 'X']

        df.loc['cums_vars_3_m1'] = [mean_cums_vars[0][0][3], 'X', mean_cums_vars[0][1][3], 'X']
        df.loc['cums_vars_3_m2'] = [mean_cums_vars[1][0][3], 'X', mean_cums_vars[1][1][3], 'X']

        print(df.round(4))
