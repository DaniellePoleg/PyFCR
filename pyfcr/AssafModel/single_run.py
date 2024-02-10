from rpy2.robjects import Formula, IntVector
from Base.utilities import *

utils = importr('utils')
base = importr('base')
survival = importr('survival')
stats = importr('stats')

utils.chooseCRANmirror(ind=1)


import math
import ._estimators as m


def run_multiple_observations(X, delta, Z, frailty_exponent, cox_weights=np.ones(shape=(500,), dtype=float), uniform=True):
    n_clusters = delta.shape[0]
    n_members = delta.shape[1]
    n_competing_risks = delta.shape[2]
    n_covariates = Z.shape[2]
    beta_coefficients_estimators = np.ndarray(shape=(n_competing_risks, n_members, n_covariates))
    cumulative_hazards_estimators = []
    hazard_at_event = np.empty(shape=(n_clusters*n_competing_risks, n_members))
    for i in range(n_members):
        try:
            cur_beta_coefficients, cur_hazard_at_event, cur_cumulative_hazard = q_single_observation(i, X, delta, Z, frailty_exponent, cox_weights, uniform=uniform)
        except:
            return beta_coefficients_estimators, hazard_at_event, cumulative_hazards_estimators
        beta_coefficients_estimators[:, i, :] = cur_beta_coefficients
        hazard_at_event[:, i] = cur_hazard_at_event
        cumulative_hazards_estimators.append(cur_cumulative_hazard)
    return beta_coefficients_estimators, hazard_at_event, cumulative_hazards_estimators


def single_run_results(beta_coefficients, frailty_covariance, cumulative_hazards, should_print=True):
    mean_beta_coefficients = beta_coefficients.mean(axis=3)
    var_beta_coefficients = beta_coefficients.var(axis=3)
    mean_cumulative_hazards = np.nanmean(cumulative_hazards, axis=3)
    var_cumulative_hazard = np.nanvar(cumulative_hazards, axis=3)
    mean_frailty_covariance = frailty_covariance.mean(axis=2)
    var_frailty_covariance = frailty_covariance.var(axis=2)

    #todo: calc confidence interval like in biometrics model
    confidence = (1 - 0.95) / 2
    confidence_interval_0_0 = np.percentile(beta_coefficients[0, 0, :], [100 * confidence, 100 * (1 - confidence)])
    confidence_interval_0_1 = np.percentile(beta_coefficients[1, 0, :], [100 * confidence, 100 * (1 - confidence)])
    confidence_interval_1_0 = np.percentile(beta_coefficients[0, 1, :], [100 * confidence, 100 * (1 - confidence)])
    confidence_interval_1_1 = np.percentile(beta_coefficients[1, 1, :], [100 * confidence, 100 * (1 - confidence)])

    if should_print:
        print("mean_beta_coefficients: ", np.round(mean_beta_coefficients, 4))
        print("emp_var_beta_coefficients: ", np.round(var_beta_coefficients, 4))
        print("mean_frailty_covariance: ", np.round(mean_frailty_covariance, 4))
        print("emp_var_frailty_covariance: ", np.round(var_frailty_covariance, 4))
        print("mean_cumulative_hazards: ", np.round(mean_cumulative_hazards, 4))
        print("emp_var_cumulative_hazard: ", np.round(var_cumulative_hazard, 4))
        print("confidence_interval_0_0", np.round(confidence_interval_0_0, 4))
        print("confidence_interval_1_0", np.round(confidence_interval_1_0, 4))
        print("confidence_interval_0_1", np.round(confidence_interval_0_1, 4))
        print("confidence_interval_1_1", np.round(confidence_interval_1_1, 4))
    # TODO: !!! print like coxph output!!!!!!!!!!!!
    return mean_beta_coefficients, mean_frailty_covariance, mean_cumulative_hazards, var_beta_coefficients, \
        var_frailty_covariance, var_cumulative_hazard, confidence_interval_0_0, confidence_interval_0_1, \
        confidence_interval_1_0, confidence_interval_1_1


def q_single_observation(i, X, delta, Z, frailty_exponent, cox_weights, uniform=True):  # runs on single observation i
    n_covariates = Z.shape[2]
    n_comp_risk = delta.shape[2]
    beta_coefficients = np.ndarray(shape=(n_comp_risk, n_covariates))
    cumulative_hazards = []
    hazard_at_event = np.ndarray(shape=(0,))

    for j in range(n_comp_risk): #for j comp risk
        cur_delta = delta[:, i, j]
        cur_x = X[:, i]
        frailty = FloatVector(np.log(frailty_exponent[:, j]))
        srv = survival.Surv(time=FloatVector(cur_x), event=IntVector(cur_delta))
        fmla_str = "srv ~ Z0"
        for z in range(1, n_covariates):
            fmla_str += " + Z" + str(z)
        fmla_str += "+ offset(frailty)"
        fmla = Formula(fmla_str)
        fmla.environment['srv'] = srv
        dataframe = {'X': FloatVector(cur_x), 'delta': IntVector(cur_delta)}
        for k in range(n_covariates):
            cur_Z = FloatVector(Z[:, :, k].reshape(-1))
            cur_Z_name = 'Z'+str(k)
            fmla.environment[cur_Z_name] = cur_Z
            dataframe[cur_Z_name] = cur_Z
        fmla.environment['frailty'] = frailty
        try:
            cox_res = survival.coxph(fmla, data=DataFrame(dataframe), weights=FloatVector(cox_weights), ties="breslow") #todo: catch here

        except:
            print("failed at run_cox_comp_risk")
            return beta_coefficients, hazard_at_event, cumulative_hazard
        cox_fit_obj = survival.coxph_detail(cox_res)
        # cur_beta_coefficients = cox_res[0][0]
        cur_beta_coefficients = list(cox_res[0])
        beta_coefficients[j, :] = cur_beta_coefficients
        hazard = cox_fit_obj[4]
        times = cox_fit_obj[0]
        if uniform:
            z_mean = Z.mean(axis=0).T
            cumulative_hazard = list(np.cumsum(hazard / np.exp(np.dot(cur_beta_coefficients, z_mean))))
        else:
            cumulative_hazard = list(np.cumsum(hazard))
        step_function = stats.stepfun(x=FloatVector(times), y=FloatVector([0] + cumulative_hazard))
        temp = numpy2ri.rpy2py(step_function(FloatVector(cur_x)))
        hazard_at_event = np.concatenate([hazard_at_event, temp])
        cumulative_hazards.append(pd.DataFrame({'x': times, 'y': cumulative_hazard}))
    return beta_coefficients, hazard_at_event, cumulative_hazards


def get_beta_Z(beta_hat, Z, n_clusters, n_members, n_competing_risks):
    beta_z = np.zeros(shape=(n_clusters, n_members, n_competing_risks))
    for i in range(n_members):
        for j in range(n_competing_risks):
            beta_z[:, i, j] = np.dot(beta_hat[j, i, :], Z[:, 0, :].T)
    return np.concatenate([beta_z[:, :, j] for j in range(n_competing_risks)])


def get_cumulative_hazard_estimators(cumulative_hazards, thresholds):
    cumulative_hazards_at_points = np.full((len(cumulative_hazards), len(cumulative_hazards[0]), len(thresholds)), np.nan)
    for i, cum in enumerate(cumulative_hazards): #i members
        for k, current_cumulative_hazard in enumerate(cum): #k comp_risk
            for j, threshold in enumerate(thresholds):
                temp = current_cumulative_hazard[current_cumulative_hazard.x > threshold]
                if not temp.empty:
                    cumulative_hazards_at_points[i, k, j] = current_cumulative_hazard.y[temp.index[0]]
    return cumulative_hazards_at_points


def run_loop(X, delta, Z, cumulative_hazard_thresholds, max_iterations=100, convergence_threshold=0.01, n_bootstrap=100, uniform=True):
    n_clusters = delta.shape[0]
    n_members = delta.shape[1]
    n_competing_risks = delta.shape[2]
    n_covariates = Z.shape[2]
    cumulative_hazards_estimators_bootstrap = np.zeros((n_members, n_competing_risks, 4, n_bootstrap), dtype=float)
    beta_coefficients_estimators_bootstrap = np.zeros((n_competing_risks, n_members, n_covariates, n_bootstrap), dtype=float)
    frailty_covariance_estimators_bootstrap = np.zeros((n_competing_risks, n_competing_risks, n_bootstrap), dtype=float)
    mu = np.zeros(n_competing_risks)
    # points_for_gauss_hermite, weights_for_gauss_hermite = get_pts_wts2(competing_risks=n_competing_risks, gh=np.polynomial.hermite.hermgauss(N_P), prune=0.2) #todo:reconsider the python implementation
    points_for_gauss_hermite, weights_for_gauss_hermite = get_pts_wts(competing_risks=n_competing_risks, gh=gauss_hermite_calculation(N_P), prune=0.2)

    for boot in range(n_bootstrap):
        frailty_exponent = np.ones((n_clusters, n_competing_risks), dtype=float)
        beta_coefficients_estimators = np.zeros((n_competing_risks, n_members, n_covariates), dtype=float)
        frailty_covariance_estimators = np.diag(np.ones(n_competing_risks))
        cumulative_hazards_estimators = None
        random_cox_weights = np.random.exponential(size=(n_clusters,))
        random_cox_weights = random_cox_weights / random_cox_weights.mean()
        iteration_cnt = 0
        convergence = 1
        while (convergence > convergence_threshold) & (iteration_cnt < max_iterations):
            old_beta_hat = beta_coefficients_estimators
            old_omega_varcov_hat = frailty_covariance_estimators
            try:
                beta_coefficients_estimators, hazard_at_event_df, cumulative_hazards_estimators = run_multiple_observations(X, delta, Z,
                                                                                                       frailty_exponent, random_cox_weights,
                                                                                                       uniform=uniform)
            except:
                print("failed in multiple observations phase")
                return beta_coefficients_estimators, frailty_covariance_estimators, cumulative_hazards_estimators
            if np.any(np.isnan(frailty_covariance_estimators)):
                print("varcov has nans!")
                return beta_coefficients_estimators, frailty_covariance_estimators, cumulative_hazards_estimators
            # beta_coefficients_estimators = np.array([[[0.5, 0.5], [0.5, 0.5]],[[2.5, 2.5], [2.5, 2.5]]])
            # beta_coefficients_estimators = np.array([[[0.5],[0.5]],[[2.5],[2.5]]])

            # frailty_covariance_estimators = np.array([[1, 0.5], [0.5, 1.5]])
            beta_Z = get_beta_Z(beta_coefficients_estimators, Z, n_clusters, n_members, n_competing_risks)
            # beta_Z = get_beta_Z2(beta_coefficients_estimators, Z, n_clusters, n_members, n_competing_risks)
            # assert np.all(beta_Z2==beta_Z)
            gauss_hermite_points, gauss_hermite_weights = multi_gauss_hermite_calculation(competing_risks=n_competing_risks, sigma=frailty_covariance_estimators, pts=points_for_gauss_hermite, wts=weights_for_gauss_hermite)  # todo: catch error

            delta_sums = delta.sum(axis=1)
            answrd = np.ones((n_clusters, n_members))
            skip = np.zeros((n_competing_risks, n_members))
            hazard = hazard_at_event_df

            frailty_covariance_estimators = m.calculate_frailty_covariance_estimators_assaf_c(delta_sums,
                                               gauss_hermite_points,
                                               gauss_hermite_weights,
                                               hazard,
                                               beta_Z,
                                               answrd,
                                               random_cox_weights,
                                               skip)

            frailty_exponent = m.calculate_frailty_exponent_estimators_assaf_C(delta_sums,
                                              gauss_hermite_points,
                                              gauss_hermite_weights,
                                              hazard,
                                              beta_Z,
                                              answrd,
                                              skip)

            convergence_betas = np.sum(np.abs(old_beta_hat - np.array(beta_coefficients_estimators)))
            convergence_frailty_covariance = np.sum(np.abs(old_omega_varcov_hat - frailty_covariance_estimators))
            convergence = convergence_betas + convergence_frailty_covariance

            if math.isnan(convergence):
                break
            iteration_cnt += 1
        beta_coefficients_estimators_bootstrap[:, :, :, boot] = beta_coefficients_estimators
        frailty_covariance_estimators_bootstrap[:, :, boot] = frailty_covariance_estimators
        cumulative_hazards_estimators_bootstrap[:, :, :, boot] = get_cumulative_hazard_estimators(cumulative_hazards_estimators, cumulative_hazard_thresholds)
    return beta_coefficients_estimators_bootstrap, frailty_covariance_estimators_bootstrap, cumulative_hazards_estimators_bootstrap

