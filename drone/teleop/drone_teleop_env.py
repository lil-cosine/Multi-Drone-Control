from typing import Any

from gymnasium.core import ObsType

from drone.drone_env import DroneEnv


class DroneTeleOpEnv(DroneEnv):
    """
    Teleoperation environment using QGroundControl/PX4.

    Motor control is handled entirely by QGroundControl via Mavlink.
    This env only processes manipulator actions (arm joints + gripper).

    Action space (3,): [joint1_pos, joint2_pos, gripper_close]
    """

    def __init__(self, drone_controller_factories=None, animal_controller_factories=None, **kwargs):
        super().__init__(
            drone_controller_factories=drone_controller_factories,
            animal_controller_factories=animal_controller_factories,
            **kwargs,
        )