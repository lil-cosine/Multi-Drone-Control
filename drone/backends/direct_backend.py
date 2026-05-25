import numpy as np

from pegasus.simulator.logic.backends import BackendConfig

from drone.drone_utils.pegasus_backend import PegasusBackend


class DirectBackend(PegasusBackend):

    def __init__(self, config=BackendConfig()):
        super().__init__(config)

        self.motor_command = np.zeros(4)

    def input_reference(self):
        return self.motor_command

    def set_motor_command(self, cmd):
        self.motor_command = np.asarray(cmd, dtype=float)