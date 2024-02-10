from read_simulated_data import get_data
from simulations_run import *
from datetime import datetime
from argparse import ArgumentParser
from pyfcr import Config, SIMULATION


def main(should_simulate_data):
    start = datetime.now().time()
    print("start time =", start)

    # run saved data
    # members = 2
    # n_clusters = 200
    # X = np.loadtxt("X.txt").reshape((n_clusters, members))
    # Z = np.loadtxt("Z.txt").reshape((n_clusters, 1))
    # delta= np.loadtxt("delta.txt")
    # delta = delta.reshape((delta.shape[0], members, delta.shape[1] // members))
    # beta_hat_coef_df, omega_varcov_hat, cumulative_hazard_df = run_loop(X, delta, Z)
    # print(beta_hat_coef_df, omega_varcov_hat)

    #
    config = get_job_config()

    if should_simulate_data:
        n_simulations = 3
        n_clusters=500
        members=2
        comp_risk=2
        # X = np.loadtxt("new_x.txt").reshape((n_clusters, members))
        # Z = np.loadtxt("new_z.txt").reshape((n_clusters, 1))
        # delta = np.loadtxt("new_delta.txt").reshape((n_clusters, members, comp_risk))
        # beta_hat_coef_df, omega_varcov_hat, cumulative_hazard_df = run_loop(X, delta, Z)
        # return beta_hat_coef_df, omega_varcov_hat
        betas, sigmas, cums_res, n_simulations = multi_sample_and_run(config)
        print_statistical_results(betas, sigmas, cums_res, n_simulations)

    #
    else:  # use Assaf simulated data from git hub
        X, Z, delta = get_data()
        beta_hat, sigma_hat, cumulative_hazard_df = run_loop(X, delta, Z)

    # print("betas: ", str(beta_hat))
    # print("sigmas: ", str(sigma_hat))

    end = datetime.now().time()
    print("end time =", end)

def get_job_config() -> Config:
    args = parse_args()
    config = Config(args.run_type)
    return config

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--run_type", dest="run_type", default=SIMULATION)
    return parser.parse_args()

if __name__ == '__main__':
    main(True)
