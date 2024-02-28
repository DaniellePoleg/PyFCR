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

        elif self.run_type == RunMode.ANALYSIS:
            self.n_covariates = config.n_covariates
            self.read_data(config.data_path)

    def read_data(self, data_path):
        raise NotImplementedError

    def simulate_data(self):
        raise NotImplementedError

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
        self.beta_coefficients_res = None
        self.frailty_covariance_res = None
        self.cumulative_hazards_res = None

    def run(self):
        self.model.simulate_data()
        self.bootstrap_run()

    def bootstrap_run(self):
        raise NotImplementedError

    def get_beta_z(self):
        raise NotImplementedError

    def get_survival_formula_and_data(self, X, frailty_exponent, cur_competing_risk):
        cur_delta = self.model.deltas[:, :, cur_competing_risk].reshape(-1)
        formula_str = "srv ~"
        for i in range(self.model.n_covariates):
            formula_str += " + Z" + str(i)
        formula_str += "+ offset(frailty)"
        formula = Formula(formula_str)
        dataframe = {'X': FloatVector(X), 'delta': IntVector(cur_delta)}
        for i in range(self.model.n_covariates):
            cur_Z = FloatVector(self.model.Z[:, :, i].reshape(-1))
            cur_Z_name = 'Z' + str(i)
            formula.environment[cur_Z_name] = cur_Z
            dataframe[cur_Z_name] = cur_Z

        srv = survival.Surv(time=FloatVector(X), event=IntVector(cur_delta))
        formula.environment['srv'] = srv

        frailty = FloatVector(np.log(frailty_exponent[:, cur_competing_risk]))
        formula.environment['frailty'] = frailty
        return formula, dataframe

    def get_cumulative_hazards(self, hazard, beta_coefficients):
        raise NotImplementedError

    def get_cox_estimators(self, frailty_exponent, cox_weights):
        raise NotImplementedError

    def get_cumulative_hazard_estimators(self):
        raise NotImplementedError

    def calculate_empirical_variance(self, estimators, mean):
        x = np.array([np.power(estimators[i, :, :] - mean, 2)
                      for i in range(self.model.n_simulations)])
        sums = np.sum(x, where=~np.isnan(x), axis=0)
        return sums / (self.model.n_simulations - 1)

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

    def get_estimators_dimensions(self, n_repeats: int) -> list:
        raise NotImplementedError

    def analyze_statistical_results(self, empirical_run: bool) -> None:
        raise NotImplementedError

    def print_summary(self) -> None:
        self.analyze_statistical_results(empirical_run=True)

    def visualize_results(self) -> None:
        visualize_results(self.beta_coefficients_res, self.frailty_covariance_res)