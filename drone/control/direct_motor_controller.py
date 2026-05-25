import numpy as np


class DirectMotorController:
    """
    Minimal controller that forwards raw motor commands directly to the drone backend.

    This controller bypasses any flight controller logic (PID, attitude estimation, etc.)
    and is intended for low-level testing or environments where motor commands are
    generated externally (e.g. by an RL policy).

    Args:
        drone: The drone instance exposing a `backend.set_motor_command()` interface.
    """

    def __init__(self, drone):
        self.drone = drone

    def post_init(self):
        """Called after the simulation environment is fully initialised. No-op for this controller."""
        pass

    def reset(self, reset_pos=None):
        """
        Reset controller state at the start of a new episode.

        Args:
            reset_pos: Ignored. Included for interface compatibility with other controllers.
        """
        pass

    def close(self):
        """Release any resources held by the controller. No-op for this controller."""
        pass

    def post_step(self, action):
        """
        Apply a motor command at the end of each simulation step.

        Args:
            action: Array-like of 4 motor thrust values in the range expected by the backend.
                    If None, all motors are set to zero.
        """

        if action is None:
            action = np.zeros(4)

        self.drone.backend.set_motor_command(action)