
from .assaf import AssafDataModel, AssafRunner, AssafMultiRunner
from .biometrics import BiometricsDataModel, BiometricsMultiRunner, BiometricsRunner
from .config import Config, RunMode, FCRType


class FCR:
    def __init__(self, config: Config):
        if config.fcr_type == FCRType.BIOMETRICS:
            self._model = BiometricsDataModel(config)
            self._runner = BiometricsMultiRunner(self._model) if config.run_type == RunMode.SIMULATION else \
                BiometricsRunner(self._model)
        elif config.fcr_type == FCRType.ASSAF:
            self._model = AssafDataModel(config)
            self._runner = AssafMultiRunner(self._model) if config.run_type == RunMode.SIMULATION else \
                AssafRunner(self._model)
        else:
            assert 1==0 #todo: send error

    def run(self):
        self._runner.run()

    def print_summary(self):
        self._runner.print_summary()