from .utilities import *
from single_run import run_loop, single_run_results
import random


def simulate_data(config):
    competing_risks = config.competing_risks
    n_clusters = config.n_clusters
    beta_coefficients = np.array(config.beta_coefficients)
    members_in_cluster = config.members_in_cluster
    frailty_mean = config.frailty_mean
    frailty_covariance = config.frailty_covariance
    uniform = config.uniform
    has_censoring = config.has_censoring
    random.seed(10)
    assert len(frailty_mean) == competing_risks
    assert len(frailty_covariance) == competing_risks
    deltas = np.empty(shape=(n_clusters, members_in_cluster, competing_risks), dtype=bool)
    event_type = np.empty((n_clusters, members_in_cluster))
    times = np.empty((n_clusters, members_in_cluster))
    n_covariates = beta_coefficients.shape[2]
    assert beta_coefficients.shape[0] == competing_risks
    Z = np.zeros((n_clusters, 1, n_covariates))

    for k in range(n_clusters):
        frailty_variates = np.random.multivariate_normal(frailty_mean, frailty_covariance)
        covariates = np.empty(shape=(n_covariates, 1), dtype=float)
        if uniform:
            for i in range(n_covariates):
                covariates[i] = np.random.uniform(0, 1)
        else:
            for i in range(n_covariates):
                covariates[i] = np.random.randint(0, 2)

        gammas = np.zeros(shape=(competing_risks, members_in_cluster))
        for j in range(competing_risks):
            for i in range(members_in_cluster):
                # if n_covariates == 1:
                beta_Z = np.dot(beta_coefficients[j, i], covariates) #todo: calc for each member
                #todo: add option for >1 covariates
                # else:
                #     beta_Z = np.dot(beta_coefficients[:, i], covariates)
                gammas[j, i] = np.exp(beta_Z + frailty_variates[j]) #todo: maybe frailty is also for each member i

        x = np.zeros(shape=(members_in_cluster, competing_risks))
        T_0 = np.zeros(shape=(members_in_cluster))
        for i in range(members_in_cluster):
            gamma = gammas[:, i].sum()
            T_0[i] = np.random.exponential(1 / gamma, 1)
            #we draw an single value from the probability, and take the index of the value that is not 0. see documentation of multinomial
            x[i, :] = np.random.multinomial(n=1, pvals=[gammas[j, i] / gamma for j in range(competing_risks)], size=1)
        J = np.zeros(members_in_cluster)
        for i in range(members_in_cluster):
            J[i] = int(np.where(x[i] == 1)[0])+1

        if has_censoring:
            censoring_time = np.random.uniform(0, 0.3, members_in_cluster)
            t = [min(T_0[i], censoring_time[i]) for i in range(members_in_cluster)]
            delta = [int(t[i] == censoring_time[i]) for i in range(members_in_cluster)]  # 1 if censored
            for i in range(members_in_cluster):
                if delta[i] == 1:
                    J[i] = 0
                    x[i] = [0] * competing_risks
        else:
            t = T_0

        event_type[k] = J
        deltas[k] = x
        times[k] = t
        Z[k, :] = covariates.reshape(-1)
    return times, Z, deltas, event_type


def sample_and_run(config):
    X, Z, delta, event_type = simulate_data(config)
    #todo:remove these lines when finish debugging
    # np.savetxt("new_x.txt", X.reshape(-1))
    # np.savetxt("new_z.txt", Z.reshape(-1))
    # np.savetxt("new_delta0.txt", delta[:,:,0].reshape(-1))
    # np.savetxt("new_delta1.txt", delta[:, :, 1].reshape(-1))
    # np.savetxt("new_delta.txt", delta.reshape(-1))
    beta_coefficients, frailty_covariance, cum_res = run_loop(X, delta, Z, config.thresholds_cumulative_hazards,
                                                              n_bootstrap=config.bootstrap, uniform=config.uniform)
    mean_beta_coefficients, mean_frailty_covariance, mean_cumulative_hazards, var_beta_coefficients, \
        var_frailty_covariance, var_cumulative_hazard, confidence_interval_0_0, confidence_interval_0_1, \
        confidence_interval_1_0, confidence_interval_1_1 = \
        single_run_results(beta_coefficients, frailty_covariance, cum_res, should_print=False)
    return mean_beta_coefficients, mean_frailty_covariance, event_type, mean_cumulative_hazards, confidence_interval_0_0, \
        confidence_interval_0_1, confidence_interval_1_0, confidence_interval_1_1,  var_beta_coefficients, var_frailty_covariance, var_cumulative_hazard


def multi_sample_and_run(config):
    competing_risks = config.n_competing_risks
    n_simulations = config.n_simulations
    n_clusters = config.n_clusters
    beta_coefficients = np.array(config.beta_coefficients)
    members_in_cluster = config.n_members_in_cluster
    calculate_event_types = config.calculate_event_types
    n_covariates = 1
    if len(beta_coefficients.shape) > 2:
        n_covariates = beta_coefficients.shape[2]
    #todo: rename and refactor all these empty lists
    cums_res = np.zeros((members_in_cluster, competing_risks, 4, n_simulations), dtype=float)
    betas = np.zeros((competing_risks, members_in_cluster, n_covariates, n_simulations), dtype=float)
    sigmas = np.zeros((competing_risks, competing_risks, n_simulations), dtype=float)
    cums_vars = np.full((members_in_cluster, competing_risks, 4, n_simulations), np.nan)
    betas_vars = np.zeros((competing_risks, members_in_cluster, n_covariates, n_simulations), dtype=float)
    sigmas_vars = np.zeros((competing_risks, competing_risks, n_simulations), dtype=float)
    censoring_rate_1 = []
    censoring_rate_2 = []
    event_type_1 = []
    event_type_2 = []
    event_type_both_1 = []
    event_type_both_2 = []
    event_type_different = []
    coverage_rates_df = pd.DataFrame(columns=range(28))

    coverage_rate_0_0 = 0
    coverage_rate_1_0 = 0
    coverage_rate_0_1 = 0
    coverage_rate_1_1 = 0
    for i in range(n_simulations):
        print("simulation number: ", i)
        beta_hat_coef_df, omega_varcov_hat, event_type, cums, conf_0_0, conf_0_1, conf_1_0, conf_1_1, beta_var, sigma_var, cums_var = sample_and_run(config)
        if calculate_event_types:
            if beta_coefficients[0] > conf_0_0[0] and beta_coefficients[0]< conf_0_0[1]:
                coverage_rate_0_0 += 1
            if beta_coefficients[0] > conf_0_1[0] and beta_coefficients[0] < conf_0_1[1]:
                coverage_rate_0_1 += 1
            if beta_coefficients[1] > conf_1_0[0] and beta_coefficients[1]< conf_1_0[1]:
                coverage_rate_1_0 += 1
            if beta_coefficients[1] > conf_1_1[0] and beta_coefficients[1] < conf_1_1[1]:
                coverage_rate_1_1 += 1
            censoring_rate_1.append(sum([int(i[0] == 0) for i in event_type]))
            censoring_rate_2.append(sum([int(i[1] == 0) for i in event_type]))
            event_type_1.append(sum([int(i[0] == 1) + int(i[1] == 1) for i in event_type]))
            event_type_2.append(sum([int(i[0] == 2) + int(i[1] == 2) for i in event_type]))
            event_type_both_1.append(sum([int(i[0] == 1 and i[1] == 1) for i in event_type]))
            event_type_both_2.append(sum([int(i[0] == 2 and i[1] == 2) for i in event_type]))
            event_type_different.append(
                sum([(int(i[0] == 1 and i[1] == 2) or int(i[0] == 2 and i[1] == 1)) for i in event_type]))
        #todo: coverage rates
        # coverage_rates_df.loc[len(coverage_rates_df)] = conf_0
        cums_res[:, :, :, i] = cums
        betas[:, :, :, i] = beta_hat_coef_df
        sigmas[:, :, i] = omega_varcov_hat
        betas_vars[:, :, :, i] = beta_var
        sigmas_vars[:, :, i] = sigma_var
        cums_vars[:, :, :, i] = cums_var

    if calculate_event_types:
        censoring = np.array(censoring_rate_1) + np.array(censoring_rate_2)
        values = [censoring, event_type_1, event_type_2, event_type_different, event_type_both_1, event_type_both_2]
        print("results for run of n clusters:", n_clusters)
        print("average:", [np.round(np.mean(value), 2) for value in values])
        print("SD:", [np.round(np.std(value), 2) for value in values])
        print("coverage rate beta 0 - 0: ", np.round(coverage_rate_0_0 / n_simulations, 2))
        print("coverage rate beta 0 - 1: ", np.round(coverage_rate_0_1 / n_simulations, 2))
        print("coverage rate beta 1 - 0: ", np.round(coverage_rate_1_0 / n_simulations, 2))
        print("coverage rate beta 1 - 1: ", np.round(coverage_rate_1_1 / n_simulations, 2))

    print("variance of betas from bootstrap: ", np.round(np.mean(betas_vars, axis=3), 4))
    print("variance of sigmas from bootstrap: ", np.round(np.mean(sigmas_vars, axis=2), 4))
    print("variance of cumulative hazards from bootstrap: ", np.round(np.nanmean(cums_vars, axis=3), 4))

    coverage_rates_df.to_csv("conf_intervals.csv")

    np.savetxt("beta_coefficient_estimators.csv", betas.reshape(-1), delimiter=",")
    np.savetxt("cumulative_hazard_estimators.csv", cums_res.reshape(-1), delimiter=",")
    np.savetxt("frailty_covariance_estimators.csv", sigmas.reshape(-1), delimiter=",")

    np.savetxt("beta_coefficient_var_boot.csv", betas_vars.reshape(-1), delimiter=",")
    np.savetxt("cumulative_hazard_var_boot.csv", cums_vars.reshape(-1), delimiter=",")
    np.savetxt("frailty_covariance_var_boot.csv", sigmas_vars.reshape(-1), delimiter=",")

    return betas, sigmas, cums_res, n_simulations

#todo: calculate_empirical_variance and use it in print method

def print_statistical_results(beta_coefficient_estimators, frailty_covariance_estimators, cumulative_hazard_estimators,
                              n_simulations):
    mean_beta_coefficients = beta_coefficient_estimators.mean(axis=3)
    mean_cumulative_hazards = np.nanmean(cumulative_hazard_estimators, axis=3) #todo: RuntimeWarning: Mean of empty slice
    mean_frailty_covariance = frailty_covariance_estimators.mean(axis=2)

    var_beta_coefficients = np.array([np.power(beta_coefficient_estimators[:, :, :, i] - mean_beta_coefficients, 2) for i in range(n_simulations)]).sum(axis=0) / (
                n_simulations - 1)
    var_cumulative_hazard = np.nansum(np.array([np.power(cumulative_hazard_estimators[:, :, :, i] - mean_cumulative_hazards, 2) for i in range(n_simulations)]), axis=0) / (
                n_simulations - 1)
    var_frailty_covariance = np.array([np.power(frailty_covariance_estimators[:, :, i] - mean_frailty_covariance, 2)
                                       for i in range(n_simulations)]).sum(axis=0) / (n_simulations - 1)

    print("mean_beta_coefficients: ", np.round(mean_beta_coefficients, 4))
    print("emp_var_beta_coefficients: ", np.round(var_beta_coefficients, 4))
    print("mean_frailty_covariance: ", np.round(mean_frailty_covariance, 4))
    print("emp_var_frailty_covariance: ", np.round(var_frailty_covariance, 4))
    print("mean_cumulative_hazards: ", np.round(mean_cumulative_hazards, 4))
    print("emp_var_cumulative_hazard: ", np.round(var_cumulative_hazard, 4))
    return mean_beta_coefficients, mean_frailty_covariance, mean_cumulative_hazards, var_beta_coefficients, \
        var_frailty_covariance, var_cumulative_hazard