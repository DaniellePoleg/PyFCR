# FRAILTY-BASED COMPETING RISKS MODEL FOR MULTIVARIATE SURVIVAL DATA 
A Python package for frailty-based multivariate survival data analysis with competing risks.

[Danielle Poleg](https://github.com/DaniellePoleg), [Malka Gorfine](https://www.tau.ac.il/~gorfinem/) 2023

## Installation
```console
pip install PyFCR
```

## Quick Start
This package has 2 run types:
### Simulation
Simulation is used to examine the model's performance. To start with a simple simulation, run the following code:
```python
from pyfcr import FCR, Config, RunMode, FCRType

if __name__ == "__main__":
    config = Config(run_type=RunMode.SIMULATION, fcr_type=FCRType.BIOMETRICS)
    fcr = FCR(config)
    fcr.run()
    fcr.print_summary()

```
To perform your own simulation with non-default parameters, add your configurations to the Config object.

### Analysis
Analysis is used to evaluate the model on an existing dataset. The dataset should consist of 3 different csv files: 
- deltas.csv - the event types data. Should be a file of size: n_clusters * n_members * n_competing_risks
- X.csv - the event times data. Should be a file of size: n_clusters * n_members * n_covariates
- Z.csv - the covariates data. Should be a file of size: n_clusters * n_members

```python
from pyfcr import FCR, Config, RunMode, FCRType

if __name__ == "__main__":
    config = Config(run_type=RunMode.ANALYSIS, fcr_type=FCRType.BIOMETRICS, data_path='<path_to_data_files>')
    fcr = FCR(config)
    fcr.run()
    fcr.print_summary()
```

## How to Contribute
1. Open Github issues to suggest new features or to report bugs\errors
2. Contact Danielle if you want to add a usage example to the documentation 
3. If you want to become a developer (thank you, we appreciate it!) - please contact Danielle for developers' on-boarding 

Danielle Poleg: daniellepoleg@gmail.com