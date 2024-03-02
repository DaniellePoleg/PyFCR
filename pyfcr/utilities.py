import pandas as pd
import numpy as np
import itertools
import warnings
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, numpy2ri, DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

survival = importr('survival')
base = importr('base')
stats = importr('stats')

N_P = 20


def hermite(points, z):
    p1 = 1 / np.pi ** 0.4
    p2 = 0
    for j in range(1, points + 1):
        p3 = p2
        p2 = p1
        p1 = z * np.sqrt(2 / j) * p2 - np.sqrt((j - 1) / j) * p3
    pp = np.sqrt(2 * points) * p2
    return p1, pp


def gauss_hermite_calculation(n, iterlim=50):
    x = np.zeros(n)
    w = np.zeros(n)
    m = (n + 1) // 2
    z = 0
    for i in range(m):
        if i == 0:
            z = np.sqrt(2 * n + 1) - 2 * (2 * n + 1) ** (-1 / 6)
        elif i == 1:
            z = z - np.sqrt(n) / z
        elif i == 2 or i == 3:
            z = 1.9 * z - 0.9 * x[i - 2]
        else:
            z = 2 * z - x[i - 2]
        j = 0
        p = (0, 0)
        for j in range(1, iterlim + 1):
            z1 = z
            p = hermite(n, z)
            z = z1 - p[0] / p[1]
            if np.abs(z - z1) <= np.exp(-15):  # todo: make sure this is how to define the decimal
                break
        if j == iterlim:
            warnings.warn("iteration limit exceeded")
        x[i] = z
        x[n - 1 - i] = -(x[i])
        w[n - 1 - i] = 2 / p[1] ** 2
        w[i] = w[n - 1 - i]
    r = pd.DataFrame({"Points": x * np.sqrt(2), "Weights": w / sum(w)})
    return r


def get_pts_wts(competing_risks, gh, prune=None):
    dm = competing_risks
    pts = create_factor_matrix(dm, gh['Points'])
    wts = create_factor_matrix(dm, gh['Weights'])
    wts = np.prod(wts, axis=1)

    if prune:
        qwt = np.quantile(wts, q=prune)
        pts = pts[wts > qwt]
        wts = wts[wts > qwt]
    return pts, wts


def get_pts_wts2(competing_risks, gh, prune=0.2):
    dm = competing_risks
    pts = create_factor_matrix(dm, gh[0][::-1])
    wts = create_factor_matrix(dm, gh[1])
    wts = np.prod(wts, axis=1)

    if prune:
        qwt = np.quantile(wts, q=prune)
        pts = pts[wts > qwt]
        wts = wts[wts > qwt]
    return pts, wts


def multi_gauss_hermite_calculation(sigma, pts, wts):
    w, v = np.linalg.eig(sigma)
    rot = np.matmul(v, np.diag(np.sqrt(w)))
    pts3 = np.matmul(rot, pts.T).T
    return pts3, list(wts)


def create_factor_matrix(n, values):
    res = []
    someList = []
    for k in range(1, n + 1):
        someList.append(values)
    for elem in itertools.product(*someList):
        res.append(list(elem))
    return np.array([np.array(xi) for xi in res])


def flat_array(arr):
    n = len(arr)
    m = len(arr[0])
    res = np.zeros(n * m)
    for i in range(n):
        for j in range(m):
            res[i * m + j] = arr[i][j]
    return res


def get_estimators_convergence(old_betas, current_betas, old_frailty_covariance, cuurent_frailty_covariance):
    convergence_betas = np.sum(np.abs(old_betas - current_betas))
    convergence_frailty_covariance = np.sum(np.abs(old_frailty_covariance - cuurent_frailty_covariance))
    return convergence_betas + convergence_frailty_covariance


def get_hazard_at_event(X, times, cumulative_hazard):
    step_function = stats.stepfun(x=FloatVector(times), y=FloatVector([0] + cumulative_hazard))
    cur_hazard_at_event = numpy2ri.rpy2py(step_function(FloatVector(X)))
    return cur_hazard_at_event


def parse_cox_estimators(formula, data, cox_weights):
    try:
        cox_res = survival.coxph(formula, data=DataFrame(data), weights=FloatVector(cox_weights), ties="breslow")
    except Exception as e:
        print("Failed in running survival.coxph, The error is: ", e)

    cur_beta_coefficients = list(cox_res[0])
    cox_fit_obj = survival.coxph_detail(cox_res)
    hazard = cox_fit_obj[4]
    times = cox_fit_obj[0]
    return cur_beta_coefficients, hazard, times


def calculate_conf_interval(estimators, n_competing_risks):
    confidence = (1 - 0.95) / 2
    dim = estimators.shape[2]
    confs = np.empty(shape=(n_competing_risks, dim, 2))
    for i in range(n_competing_risks):
        for j in range(dim):
            confs[i, j, :] = np.percentile(estimators[:, i, j], [100 * confidence, 100 * (1 - confidence)])
    return confs


def plot_event_occurence(X, event_types):
    # Create a figure with two subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(18, 10), sharey=True)

    # Plot for 'type' 0
    df_0 = pd.DataFrame({'time': X[:, 0], 'type': event_types[:, 0]})
    dummies_0 = pd.get_dummies(df_0['type'], prefix='J')
    df_0 = pd.concat([df_0, dummies_0], axis=1)
    df_0['time_bins'] = pd.cut(df_0['time'], bins=20)
    df_0 = df_0[['time_bins', 'J_0', 'J_1', 'J_2']]
    df_0.groupby(['time_bins']).sum().plot(ax=ax0, kind='bar', width=0.8, legend=True)
    ax0.tick_params(axis='both', which='major', labelsize=10)
    ax0.tick_params(axis='both', which='minor', labelsize=10)
    ax0.set_xlabel("Time", fontdict={'size': 18})
    ax0.set_ylabel("Number of Observations", fontdict={'size': 18})
    ax0.set_title('Member 1', fontdict={'size': 18})

    # Plot for 'type' 1
    df_1 = pd.DataFrame({'time': X[:, 1], 'type': event_types[:, 1]})
    dummies_1 = pd.get_dummies(df_1['type'], prefix='J')
    df_1 = pd.concat([df_1, dummies_1], axis=1)
    df_1['time_bins'] = pd.cut(df_1['time'], bins=20)
    df_1 = df_1[['time_bins', 'J_0', 'J_1', 'J_2']]
    df_1.groupby(['time_bins']).sum().plot(ax=ax1, kind='bar', width=0.8)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.tick_params(axis='both', which='minor', labelsize=10)
    ax1.set_xlabel("Time", fontdict={'size': 18})
    ax1.set_ylabel("Number of Observations", fontdict={'size': 18})
    ax1.set_title('Member 2', fontdict={'size': 18})

    plt.tight_layout()
    fig.savefig('event distribution', dpi=300)


def visualize_results(beta_coefficients_res, frailty_covariance_res) -> None:
    a = beta_coefficients_res[0, :, :].reshape(-1)
    b = beta_coefficients_res[1, :, :].reshape(-1)
    c = frailty_covariance_res[0, 0, :]
    d = frailty_covariance_res[1, 1, :]
    e = frailty_covariance_res[0, 1, :]
    data = pd.DataFrame(np.array([a, b, c, d, e]).T, columns=['beta_1', 'beta_2', 'sigma_1', 'sigma_2', 'corr_1_2'])
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(7)
    data.boxplot(grid=True)
    plt.xlabel('Coefficient Names')
    plt.ylabel('Estimated Coefficient Values')
    red_patch = mpatches.Patch(color='red', label='Real Value')
    blue_patch = mpatches.Patch(color='green', label='Median')
    plt.legend(handles=[red_patch, blue_patch], loc='upper center')
    real_values_for_table = [0.5, 2.5, 1, 1.5, 0.5, 0.10, 0.15, 0.20, 0.25, 0.10, 0.15, 0.20, 0.25]
    for i in range(data.shape[1]):
        plt.scatter(x=i + 1, y=real_values_for_table[i], c='red')
    plt.savefig("boxplot.jpg")