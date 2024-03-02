from enum import Enum
from pathlib import Path
from typing import Dict, Callable

import numpy as np


class RunMode(Enum):
    SIMULATION = "simulation",
    ANALYSIS = "analysis"


class FCRType(Enum):
    BIOMETRICS = 1,
    ASSAF = 2


class Config:
    """
    This class is used to access configuration.
    """

    def __init__(self,
                 run_type: RunMode = RunMode.SIMULATION,
                 fcr_type: FCRType = FCRType.ASSAF,
                 n_members_in_cluster: int = 2,
                 n_competing_risks: int = 2,
                 n_clusters: int = 500,
                 uniform: bool = True,
                 n_bootstrap: int = 3,
                 thresholds_cumulative_hazards: list = [0.1, 0.15, 0.20, 0.25],
                 calculate_event_types: bool = False,
                 n_simulations: int = 3,
                 beta_coefficients: list = None,
                 frailty_mean: list = [0, 0],
                 frailty_covariance: list = [[1, 0.5], [0.5, 1.5]],
                 censoring_method: Callable = lambda x: np.random.uniform(0, 0.3, x),
                 n_covariates: int = None,
                 data_path: Path = None):
        '''
        initialize the configurations for your run
        :param run_type: RunMode.SIMULATION for simulation, RunMode.ANALYSIS for analysis
        :param fcr_type: FCRType.BIOMETRICS for Approach A, FCRType.Assaf for Approach B
        :param n_members_in_cluster: number of members within each cluster
        :param n_competing_risks: number of competing risk
        :param n_clusters: number of clusters
        :param uniform: if the distribution of covariates is uniform
        :param n_bootstrap: number of bootstrap iterations
        :param thresholds_cumulative_hazards: times for evaluation of cumulative hazard function
        :param calculate_event_types: if the output should contain event types
        :param n_simulations: number of simulations (simulation mode only)
        :param beta_coefficients: regression coefficients (simulation mode only)
        :param frailty_mean: the mean of the multivariate normal frailty model (simulation mode only)
        :param frailty_covariance: the variance of the multivariate normal frailty model (simulation mode only)
        :param censoring_method: the censoring method (simulation mode only)
        :param n_covariates: the number of covariates (analysis mode only)
        :param data_path: the path to the data to analyse (analysis mode only)
        '''
        self.run_type = run_type
        self.fcr_type = fcr_type
        self.n_members_in_cluster = n_members_in_cluster
        self.n_competing_risks = n_competing_risks
        self.n_clusters = n_clusters
        self.uniform = uniform
        self.bootstrap = n_bootstrap
        self.thresholds_cumulative_hazards = thresholds_cumulative_hazards
        self.calculate_event_types = calculate_event_types

        if self.run_type == RunMode.SIMULATION:
            self.n_simulations = n_simulations
            self.beta_coefficients = beta_coefficients if beta_coefficients else ([[[0.5],[0.5]],[[2.5],[2.5]]] if fcr_type==FCRType.ASSAF else [0.5, 2.5])
            self.frailty_mean = frailty_mean
            self.frailty_covariance = frailty_covariance
            self.censoring_method = censoring_method

        elif self.run_type == RunMode.ANALYSIS.value[0]:
            self.n_covariates = n_covariates
            self.data_path = data_path

    @staticmethod
    def print_configs(configs: Dict[str, object]) -> None:
        for config, value in configs.items():
            print(config, ": ", value)
