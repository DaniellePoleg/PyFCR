from pyfcr import FCR, Config, RunMode, FCRType

if __name__ == "__main__":
    config = Config(run_type=RunMode.SIMULATION, fcr_type=FCRType.ASSAF, beta_coefficients=[[[0.5],[0.5],[0.5]],[[2.5],[2.5],[2.5]]], n_members_in_cluster=3)
    a = FCR(config)
    a.run()
    a.print_summary()
