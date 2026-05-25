import numpy as np

from scipy.spatial.transform import Rotation as R


class PID:
    """
    Discrete-time PID controller with integral clamping.

    Args:
        kp (float):     Proportional gain.
        ki (float):     Integral gain.
        kd (float):     Derivative gain.
        dt (float):     Time step in seconds.
        limit (float):  Optional symmetric output clamp (±limit). Pass None to disable.
    """

    def __init__(self, kp, ki, kd, dt, limit=None):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.dt = dt

        self._i = 0.0       # Accumulated integral term
        self._prev = 0.0    # Previous error for derivative calculation

        self.limit = limit

    def reset(self):
        """Reset integral accumulator and previous error to zero."""

        self._i = 0.0
        self._prev = 0.0

    def update(self, e):
        """
        Compute the PID output for a given error signal.

        The integral term is clamped to [-0.3, 0.3] to prevent windup.

        Args:
            e (float): Current error (setpoint − measured value).

        Returns:
            float: Controller output, optionally clamped to ±limit.
        """

        self._i += e * self.dt
        self._i = np.clip(self._i, -0.3, 0.3)

        d = (e - self._prev) / self.dt
        self._prev = e

        out = (
            self.kp * e
            + self.ki * self._i
            + self.kd * d
        )

        if self.limit is not None:
            out = np.clip(out, -self.limit, self.limit)

        return out


# ============================================================
# Quad Mixer
# ============================================================

def mixer(throttle, roll, pitch):
    """
    Convert throttle/roll/pitch commands into individual motor thrust values
    for a standard X-configuration quadrotor.

    Motor layout (top-down view):

        M3 (FL) ------- M1 (FR)
             \\         /
              \\       /
               [body]
              /       \\
             /         \\
        M4 (RL) ------- M2 (RR)

    Mixing matrix:
        M0 = throttle - pitch - roll   (front-right)
        M1 = throttle + pitch + roll   (back-left  — note: index ordering follows array)
        M2 = throttle - pitch + roll
        M3 = throttle + pitch - roll

    Args:
        throttle (float): Collective thrust command.
        roll     (float): Roll correction command (positive = roll right).
        pitch    (float): Pitch correction command (positive = pitch forward).

    Returns:
        np.ndarray: Motor commands of shape (4,).
    """

    return np.array([
        throttle - pitch - roll,
        throttle + pitch + roll,
        throttle - pitch + roll,
        throttle + pitch - roll,
    ], dtype=np.float64)


# ============================================================
# Hover PID Controller
# ============================================================

class HoverPIDController:
    """
    Dual-loop PID controller that holds the drone at a fixed altitude while
    keeping roll and pitch at zero (level hover).

    Architecture:
        - Outer loop: altitude error → throttle correction via ``_hover_pid``.
        - Inner loop: roll/pitch error → differential motor trim via ``_att_pid``.
        - Both outputs are combined in the quad mixer and clipped to safe motor limits.

    Class-level tuning constants can be overridden in a subclass without
    changing the constructor signature.

    Args:
        drone:      Drone instance with a ``backend.set_motor_command()`` interface
                    and a ``drone._state`` attribute exposing position and attitude.
        target_z:   Desired hover altitude in metres (default: 1.0 m).
    """

    # Base throttle that approximately counteracts gravity at hover
    BASE_HOVER = 450.0

    # Altitude PID gains
    HOVER_KP = 90.0
    HOVER_KI = 0.0
    HOVER_KD = 55.0
    HOVER_LIMIT = 200.0

    # Attitude (roll/pitch) PID gains
    ATT_KP = 120.0
    ATT_KI = 0.0
    ATT_KD = 40.0
    ATT_LIMIT = 120.0

    # Motor output clamp
    MOTOR_MIN = 0.0
    MOTOR_MAX = 1100.0

    # Control loop time step (must match simulation step rate)
    DT = 0.02   # 50 Hz

    def __init__(
        self,
        drone,
        target_z=1.0,
    ):
        self.drone = drone

        self.target_z = float(target_z)

        self._hover_pid = PID(
            self.HOVER_KP,
            self.HOVER_KI,
            self.HOVER_KD,
            self.DT,
            limit=self.HOVER_LIMIT,
        )

        self._att_pid = PID(
            self.ATT_KP,
            self.ATT_KI,
            self.ATT_KD,
            self.DT,
            limit=self.ATT_LIMIT,
        )

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def post_init(self):
        """Called after the simulation environment is fully initialised. No-op for this controller."""
        pass

    def close(self):
        """Release any resources held by the controller. No-op for this controller."""
        pass

    def reset(self, reset_pos=None):
        """
        Reset both PID controllers at the start of a new episode.

        Args:
            reset_pos: Ignored. Included for interface compatibility with other controllers.
        """

        self._hover_pid.reset()
        self._att_pid.reset()

    # --------------------------------------------------------
    # Main Update
    # --------------------------------------------------------

    def post_step(self, action=None):
        """
        Compute and send motor commands at the end of each simulation step.

        Reads the current altitude and attitude from the drone state, runs both
        PID loops, mixes the outputs into per-motor commands, clips them to
        safe limits, and forwards the result to the backend.

        Args:
            action: Unused. Included for interface compatibility with other controllers.
        """

        state = self.drone.drone._state

        z = float(state.position[2])

        roll, pitch, _ = R.from_quat(
            state.attitude
        ).as_euler("xyz")

        # ----------------------------------------------------
        # Altitude PID
        # ----------------------------------------------------

        throttle = (
            self.BASE_HOVER
            + self._hover_pid.update(
                self.target_z - z
            )
        )

        # ----------------------------------------------------
        # Attitude PID
        # ----------------------------------------------------

        roll_cmd = self._att_pid.update(-roll)
        pitch_cmd = self._att_pid.update(-pitch)

        # ----------------------------------------------------
        # Mix
        # ----------------------------------------------------

        motors = mixer(
            throttle,
            roll_cmd,
            pitch_cmd,
        )

        motors = np.clip(
            motors,
            self.MOTOR_MIN,
            self.MOTOR_MAX,
        )

        # ----------------------------------------------------
        # Send To Backend
        # ----------------------------------------------------

        self.drone.backend.set_motor_command(
            motors
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def set_target_z(self, z):
        """
        Update the desired hover altitude.

        Args:
            z (float): New target altitude in metres.
        """
        self.target_z = float(z)