from pyfcr import FCR, Config, RunMode, FCRType

if __name__ == "__main__":
    config = Config(run_type=RunMode.SIMULATION, fcr_type=FCRType.BIOMETRICS)
    a = FCR(config)
    a.run()
    a.print_summary()
