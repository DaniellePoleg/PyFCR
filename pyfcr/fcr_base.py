import os
import math

from .utilities import *
from .config import Config, RunMode
from rpy2.robjects import Formula, IntVector, FloatVector
survival = importr('survival')


class FCRDataModel:

    def __init__(self, config: Config):
        self.X = None
        self.Z = None
        self.deltas = None
        self.event_types = None

        self.run_type = config.run_type
        self.n_clusters = config.n_clusters
        self.n_members = config.n_members_in_cluster
        self.n_competing_risks = config.n_competing_risks
        self.cumulative_hazard_thresholds = config.thresholds_cumulative_hazards
        self.n_threshold_cum_hazard = len(self.cumulative_hazard_thresholds)
        self.n_bootstrap = config.bootstrap
        self.uniform = config.uniform
        self.n_covariates = 1

        if self.run_type == RunMode.SIMULATION:
            self.frailty_mean = config.frailty_mean
            self.frailty_covariance = config.frailty_covariance
            self.censoring_method = config.censoring_method
            self.n_simulations = config.n_simulations
            self.calculate_event_types = config.calculate_event_types
            self.beta_coefficients = np.array(config.beta_coefficients)
            if len(self.beta_coefficients.shape) > 1:
                self.n_covariates = self.beta_coefficients.shape[self.get_n_covariates_from_beta()]
                assert self.beta_coefficients.shape[0] == self.n_competing_risks
            else:
                assert self.beta_coefficients.shape[0] == self.n_competing_risks

            assert len(self.frailty_mean) == self.n_competing_risks
            assert len(self.frailty_covariance) == self.n_competing_risks

            self.deltas = np.empty(shape=(self.n_clusters, self.n_members, self.n_competing_risks), dtype=bool)
            self.event_types = np.empty(shape=(self.n_clusters, self.n_members), dtype=int)
            self.X = np.empty(shape=(self.n_clusters, self.n_members), dtype=float)
            self.Z = np.empty(self.get_z_dimension(), dtype=float)


        elif self.run_type == RunMode.ANALYSIS:
            self.n_covariates = config.n_covariates
            self.read_data(config.data_path)

    def get_n_covariates_from_beta(self):
        raise NotImplementedError

    def get_z_dimension(self):
        raise NotImplementedError

    def read_data(self, data_path):
        self.deltas = np.loadtxt(os.path.join(data_path, "deltas.csv"), delimiter=',').reshape(
            (self.n_clusters, self.n_members, self.n_competing_risks))
        self.Z = np.loadtxt(os.path.join(data_path, "Z.csv"), delimiter=',').reshape(
            self.get_z_dimension())
        self.X = np.loadtxt(os.path.join(data_path, "X.csv"), delimiter=',').reshape((self.n_clusters, self.n_members))

    def simulate_data(self):
        raise NotImplementedError

    def get_covariates_single_simulation(self):
        covariates = np.empty(shape=(self.n_covariates, 1), dtype=float)
        if self.uniform:
            for i in range(self.n_covariates):
                covariates[i] = np.random.uniform(0, 1)
        else:
            for i in range(self.n_covariates):
                covariates[i] = np.random.randint(0, 2)
        return covariates

    def plot_event_occurence(self):
        plot_event_occurence(self.X, self.event_types)


class Runner:
    def __init__(self, dataModel):
        self.model = dataModel
        self.max_iterations = 100
        self.convergence_threshold = 0.01
        self.points_for_gauss_hermite, self.weights_for_gauss_hermite = get_pts_wts(
            competing_risks=self.model.n_competing_risks,
            gh=gauss_hermite_calculation(N_P), prune=0.2)
        self.frailty_exponent = np.ones((self.model.n_clusters, self.model.n_competing_risks), dtype=float)
        self.frailty_covariance_estimators = np.diag(np.ones(self.model.n_competing_risks))
        self.cumulative_hazards_estimators = []
        self.estimators_df = pd.DataFrame(columns=['betas', 'frailty_covariance', 'cumulative_hazards'])
        self.beta_coefficients_res = None
        self.frailty_covariance_res = None
        self.cumulative_hazards_res = None

    def run(self):
        self.model.simulate_data()
        self.bootstrap_run()

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
                    self.get_cox_estimators()
            except Exception as e:
                print("Failed in get_cox_estimators in run_single_estimation, The error is: ", e)
            self.get_frailty_estimators(hazard_at_event)
            convergence = get_estimators_convergence(old_betas, self.beta_coefficients_estimators,
                                                     old_frailty_covariance, self.frailty_covariance_estimators)
            if math.isnan(convergence):
                break
            iteration_cnt += 1

    def get_beta_z(self):
        raise NotImplementedError

    def get_frailty_estimators_arguments(self):
        delta_sums = self.get_delta_sums()
        gauss_hermite_points, gauss_hermite_weights = multi_gauss_hermite_calculation(
            sigma=self.frailty_covariance_estimators, pts=self.points_for_gauss_hermite,
            wts=self.weights_for_gauss_hermite)
        beta_z = self.get_beta_z()
        return delta_sums, gauss_hermite_points, gauss_hermite_weights, beta_z

    def calculate_frailty_covariance_estimators(self, args):
        raise NotImplementedError

    def calculate_frailty_exponent_estimators(self, args):
        raise NotImplementedError

    def get_frailty_estimators(self, hazard_at_event):
        rowSums, gauss_hermite_points, gauss_hermite_weights, beta_z = self.get_frailty_estimators_arguments()
        self.frailty_covariance_estimators = self.calculate_frailty_covariance_estimators(
            [rowSums, gauss_hermite_points, gauss_hermite_weights, hazard_at_event, beta_z, self.random_cox_weights])
        self.frailty_exponent = self.calculate_frailty_exponent_estimators([rowSums, gauss_hermite_points,
                                                                            gauss_hermite_weights, hazard_at_event,
                                                                            beta_z])

    def get_delta_sums(self):
        return NotImplementedError

    def get_survival_formula_and_data(self, X, cur_delta, frailty_exponent, cur_competing_risk):
        formula_str = "srv ~"
        for z in range(self.model.n_covariates):
            formula_str += " + Z" + str(z)
        formula_str += "+ offset(frailty)"
        formula = Formula(formula_str)

        dataframe = {'X': FloatVector(X), 'delta': IntVector(cur_delta)}
        srv = survival.Surv(time=FloatVector(X), event=IntVector(cur_delta))
        formula.environment['srv'] = srv

        frailty = FloatVector(np.log(frailty_exponent[:, cur_competing_risk]))
        formula.environment['frailty'] = frailty

        return self.parse_covariates_fmla(formula, dataframe)

    def parse_covariates_fmla(self, formula, dataframe):
        for i in range(self.model.n_covariates):
            cur_Z = FloatVector(self.model.Z[:, :, i].reshape(-1))
            cur_Z_name = 'Z' + str(i)
            formula.environment[cur_Z_name] = cur_Z
            dataframe[cur_Z_name] = cur_Z
        return formula, dataframe

    def get_cumulative_hazards(self, hazard, beta_coefficients):
        if self.model.uniform:
            z_mean = self.get_z_mean()
            cumulative_hazard = list(np.cumsum(hazard / np.exp(np.dot(beta_coefficients, z_mean))))
        else:
            cumulative_hazard = list(np.cumsum(hazard))
        return cumulative_hazard

    def get_z_mean(self):
        raise NotImplementedError

    def get_cox_estimators(self):
        raise NotImplementedError

    def get_cumulative_hazard_estimators(self):
        raise NotImplementedError

    def calculate_empirical_variance(self, estimators, mean):
        x = np.array([np.power(estimators[i, :, :] - mean, 2)
                      for i in range(self.model.n_simulations)])
        sums = np.sum(x, where=~np.isnan(x), axis=0)
        return sums / (self.model.n_simulations - 1)

    def print_estimators(self, df, mean, standard_error, standard_deviation, getter):
        for i in range(self.model.n_competing_risks):
            df.loc[f'competing_risk_{i + 1}_param'] = getter(mean, i)
            df.loc[f'competing_risk_{i + 1}_SE'] = getter(standard_error, i)
            if standard_deviation is not None:
                df.loc[f'competing_risk_{i + 1}_SD'] = getter(standard_deviation, i)
        home_directory = os.path.expanduser("~")
        directory = os.path.join(home_directory, "pyfcr_results")
        os.makedirs(directory, exist_ok=True)
        file_name = df.columns[0].split('_')[0]
        df.to_csv(os.path.join(directory, f"{file_name}.csv"))
        print(df.round(4))

    def print_betas(self, mean, standard_error, standard_deviation) -> None:
        raise NotImplementedError

    def print_cumulative_hazards(self, mean, standard_error, standard_deviation) -> None:
        raise NotImplementedError

    def print_frailty_covariance(self, mean, standard_error, standard_deviation) -> None:
        df = pd.DataFrame(columns=[f'frailty_covariance_comp_risk_{i}_comp_risk_{j}' for i in
                                   range(1, self.model.n_competing_risks + 1) for j in
                                   range(1, self.model.n_competing_risks + 1)
                                   if i <= j])
        data = {'param': mean, 'SE': standard_error}
        if standard_deviation is not None:
            data['SD'] = standard_deviation
        for key, value in data.items():
            df.loc[key] = [value[i][j] for i in range(self.model.n_competing_risks) for j in
                           range(self.model.n_competing_risks) if i <= j]

        home_directory = os.path.expanduser("~")
        directory = os.path.join(home_directory, "pyfcr_results")
        os.makedirs(directory, exist_ok=True)
        df.to_csv(os.path.join(directory, "frailty_covariance.csv"))
        print(df.round(4))

    def analyze_statistical_results(self, empirical_run=True, multi_estimators_df=None):
        mean_betas_vars = None
        mean_sigmas_vars = None
        mean_cums_vars = None

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
            if multi_estimators_df is not None:
                betas_vars, sigmas_vars, cums_vars = self.reshape_estimators_from_df(
                    multi_estimators_df.iloc[:, 3:6],
                    self.model.n_simulations)
                mean_betas_vars = np.mean(betas_vars, axis=0)
                mean_sigmas_vars = np.mean(sigmas_vars, axis=0)
                mean_cums_vars = np.nanmean(cums_vars, axis=0)

            pd.set_option('display.max_columns', None)
            self.print_betas(mean_beta_coefficients, var_beta_coefficients, mean_betas_vars)
            self.print_frailty_covariance(mean_frailty_covariance, var_frailty_covariance, mean_sigmas_vars)
            self.print_cumulative_hazards(mean_cumulative_hazards, var_cumulative_hazard, mean_cums_vars)

        else:
            var_beta_coefficients = self.beta_coefficients_res.var(axis=axis)
            var_frailty_covariance = self.frailty_covariance_res.var(axis=axis)
            var_cumulative_hazard = np.nanvar(self.cumulative_hazards_res, axis=axis)

        return mean_beta_coefficients, mean_frailty_covariance, mean_cumulative_hazards, var_beta_coefficients, \
            var_frailty_covariance, var_cumulative_hazard

    # todo: this is not used in Assaf
    def get_multiple_confidence_intervals(self):
        a = calculate_conf_interval(self.beta_coefficients_res, self.model.n_competing_risks)
        b = calculate_conf_interval(self.frailty_covariance_res, self.model.n_competing_risks)
        c = calculate_conf_interval(self.cumulative_hazards_res, self.model.n_competing_risks)
        conf_int = np.concatenate([a.reshape(-1), b.reshape(-1), c.reshape(-1)])
        return conf_int

    def reshape_estimators_from_df(self, estimators_df, n_repeats):
        shapes = self.get_estimators_dimensions(n_repeats)
        beta_coefficients_res = np.vstack(estimators_df.iloc[:, 0]).reshape(shapes[0])
        frailty_covariance_res = np.vstack(estimators_df.iloc[:, 1]).reshape(shapes[1])
        cumulative_hazards_res = np.vstack(estimators_df.iloc[:, 2]).reshape(shapes[2])
        return beta_coefficients_res, frailty_covariance_res, cumulative_hazards_res

    def get_estimators_dimensions(self, n_repeats: int) -> None:
        raise NotImplementedError

    def print_summary(self) -> None:
        self.analyze_statistical_results(empirical_run=True, multi_estimators_df=None)

    def visualize_results(self) -> None:
        visualize_results(self.beta_coefficients_res, self.frailty_covariance_res)
