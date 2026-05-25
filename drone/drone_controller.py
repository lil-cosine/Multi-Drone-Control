import os

import numpy as np

from scipy.spatial.transform import Rotation

from pegasus.simulator.logic import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import (
    Multirotor,
    MultirotorConfig,
)

from sim.controller import Controller
from utils import get_project_path


class DroneController(Controller):

    DEFAULT_INIT_POS = np.array([0.0, 0.0, 1.0])
    DEFAULT_INIT_ROT = np.array([0.0, 0.0, 0.0])

    def __init__(
        self,
        parent_env,
        backend,
        controller=None,
        stage_prefix="/World/drone",
        init_pos=None,
        init_rot=None,
    ):
        self.world = parent_env.world

        self.pg = PegasusInterface()
        self.pg._world = self.world

        self.backend = backend
        self.controller = controller

        self.stage_prefix = stage_prefix

        init_pos = (
            self.DEFAULT_INIT_POS
            if init_pos is None
            else np.array(init_pos)
        )

        init_rot = (
            self.DEFAULT_INIT_ROT
            if init_rot is None
            else np.array(init_rot)
        )

        config_multirotor = MultirotorConfig()

        if backend is not None:

            if hasattr(backend, "backend"):
                config_multirotor.backends = [backend.backend]
            else:
                config_multirotor.backends = [backend]

        self.drone = Multirotor(
            self.stage_prefix,
            os.path.join(
                get_project_path(),
                "assets",
                "drones",
                "iris.usd",
            ),
            0,
            init_pos,
            Rotation.from_euler(
                "XYZ",
                init_rot,
                degrees=True,
            ).as_quat(),
            config=config_multirotor,
        )

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def post_init(self):

        if self.backend:
            self.backend.start()

        if self.controller:
            self.controller.post_init()

    def post_step(self, action=None):

        if self.controller:
            self.controller.post_step(action)

    def reset(self, reset_pos=None):

        if self.controller:
            self.controller.reset(reset_pos)

    def close(self):

        if self.controller:
            self.controller.close()

        if self.backend:
            self.backend.stop()

    # ------------------------------------------------------------
    # Runtime Controller Switching
    # ------------------------------------------------------------

    def set_controller(self, controller):

        if self.controller:
            self.controller.close()

        self.controller = controller

        if self.controller:
            self.controller.post_init()