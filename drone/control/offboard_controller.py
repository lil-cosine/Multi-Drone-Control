import time
import numpy as np

from pymavlink import mavutil


# Typemask that tells PX4 to use only position + yaw, ignoring velocity,
# acceleration, and yaw rate fields in the setpoint message.
_USE_POSITION = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

# Number of simulation steps to wait before attempting the MAVLink connection.
# This gives the simulator time to fully initialise before any network traffic is sent.
_PRIME_START_STEPS = 500

# Steps spent streaming setpoints before requesting mode switch.
# PX4 requires a stream of setpoints to already be arriving before it will
# accept the OFFBOARD mode command.
_PRIME_STEPS = 200

# Steps spent requesting OFFBOARD mode before proceeding to arm.
_MODE_SWITCH_STEPS = 100

# Steps spent sending arm commands before the controller is considered active.
_ARM_STEPS = 100


class OffboardController:
    """
    PX4 offboard position controller that drives a simulated drone to a fixed
    ENU (East-North-Up) target via MAVLink UDP.

    Startup sequence
    ----------------
    PX4 will reject an OFFBOARD mode switch unless setpoints are already being
    streamed. The controller therefore works through the following states before
    it begins active flight:

    1. **wait**    – idle until ``_PRIME_START_STEPS`` have elapsed.
    2. **connect** – open a MAVLink UDP link and wait for a heartbeat.
    3. **prime**   – stream position setpoints for ``_PRIME_STEPS`` steps.
    4. **mode**    – send the OFFBOARD mode command for ``_MODE_SWITCH_STEPS`` steps.
    5. **arm**     – send the ARM command for ``_ARM_STEPS`` steps.
    6. **active**  – continuously stream the target setpoint.

    Network topology
    ----------------
    Each vehicle is assumed to listen on UDP port ``14580 + vehicle_id``.  The
    controller binds as a *udpout* client (i.e. it sends datagrams; PX4 must be
    listening on the corresponding port).

    Args:
        drone:           Drone instance with a ``backend.vehicle_id`` attribute.
        target_pos:      Desired ENU position as ``(x, y, z)`` in metres (default: ``(0, 0, 1)``).
        target_yaw_deg:  Desired yaw in degrees, measured from East (default: ``0``).
    """

    # ---- Internal state machine labels ----
    _STATE_WAIT    = "wait"
    _STATE_CONNECT = "connect"
    _STATE_PRIME   = "prime"
    _STATE_MODE    = "mode"
    _STATE_ARM     = "arm"
    _STATE_ACTIVE  = "active"

    def __init__(
        self,
        drone,
        target_pos=(0.0, 0.0, 1.0),
        target_yaw_deg=0.0,
    ):
        self.drone = drone

        self.target_pos = np.array(target_pos, dtype=float)
        self.target_yaw_deg = float(target_yaw_deg)

        self._step = 0          # Total steps since last reset
        self._state_step = 0    # Steps elapsed in the current state
        self._state = self._STATE_WAIT

        self._offboard_conn = None  # mavutil connection object
        self._connected = False     # True once the heartbeat handshake completes

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def post_init(self):
        """Called after the simulation environment is fully initialised. No-op for this controller."""
        pass

    def reset(self, reset_pos=None):
        """
        Reset the state machine and step counters at the start of a new episode.

        Note: The existing MAVLink connection is intentionally preserved across
        resets to avoid reconnection overhead on episode boundaries.

        Args:
            reset_pos: Ignored. Included for interface compatibility with other controllers.
        """

        self._step = 0
        self._state_step = 0
        self._state = self._STATE_WAIT

    def close(self):
        """Close the MAVLink UDP connection and release associated resources."""

        if self._offboard_conn is not None:
            try:
                self._offboard_conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------

    def post_step(self, action=None):
        """
        Advance the state machine by one simulation step.

        Should be called once per simulation tick. The ``action`` argument is
        unused; control is driven entirely by the internal state machine and the
        current target position.

        Args:
            action: Unused. Included for interface compatibility with other controllers.
        """

        self._step += 1
        self._state_step += 1

        self._run_state_machine()

    # ------------------------------------------------------------
    # State Machine
    # ------------------------------------------------------------

    def _run_state_machine(self):
        """Execute the current state and handle transitions."""

        vid = self._vehicle_id()

        if self._state == self._STATE_WAIT:

            if self._step >= _PRIME_START_STEPS:

                print(f"[Offboard v{vid}] Connecting MAVLink...")

                self._transition(self._STATE_CONNECT)

        elif self._state == self._STATE_CONNECT:

            if not self._connected:
                self._connect_offboard_link()

            if self._connected:

                print(f"[Offboard v{vid}] Priming setpoints...")

                self._transition(self._STATE_PRIME)

        elif self._state == self._STATE_PRIME:

            self._send_position_setpoint()

            if self._state_step >= _PRIME_STEPS:

                print(f"[Offboard v{vid}] Requesting OFFBOARD...")

                self._transition(self._STATE_MODE)

        elif self._state == self._STATE_MODE:

            self._send_position_setpoint()
            self._set_mode_offboard()

            if self._state_step >= _MODE_SWITCH_STEPS:

                print(f"[Offboard v{vid}] Arming...")

                self._transition(self._STATE_ARM)

        elif self._state == self._STATE_ARM:

            self._send_position_setpoint()
            self._send_arm()

            if self._state_step >= _ARM_STEPS:

                print(f"[Offboard v{vid}] Active.")

                self._transition(self._STATE_ACTIVE)

        elif self._state == self._STATE_ACTIVE:

            self._send_position_setpoint()

    def _transition(self, new_state):
        """
        Move to a new state and reset the per-state step counter.

        Args:
            new_state (str): One of the ``_STATE_*`` class constants.
        """

        self._state = new_state
        self._state_step = 0

    # ------------------------------------------------------------
    # MAVLink
    # ------------------------------------------------------------

    def _vehicle_id(self):
        """Return the integer vehicle ID from the drone backend."""
        return self.drone.backend.vehicle_id

    def _target_system(self):
        """
        Return the MAVLink system ID of the target vehicle.

        PX4 uses ``vehicle_id + 1`` as its system ID by convention.
        """
        return self._vehicle_id() + 1

    def _offboard_port(self):
        """
        Return the UDP port for the offboard MAVLink link.

        Each vehicle listens on ``14580 + vehicle_id`` to avoid port collisions
        in multi-vehicle simulations.
        """
        return 14580 + self._vehicle_id()

    def _timestamp_ms(self):
        """
        Return the current Unix time in milliseconds, masked to a 32-bit integer.

        Used as the ``time_boot_ms`` field in MAVLink messages.
        """
        return int(time.time() * 1000) & 0xFFFFFFFF

    def _connect_offboard_link(self):
        """
        Open a MAVLink UDP connection to the vehicle and wait for a heartbeat.

        On success, sets ``self._connected = True``.
        On failure, logs the error and leaves ``self._connected = False`` so
        the caller can retry on the next step.
        """

        try:

            port = self._offboard_port()

            self._offboard_conn = mavutil.mavlink_connection(
                f"udpout:127.0.0.1:{port}",
                source_system=200 + self._vehicle_id(),
            )

            self._offboard_conn.wait_heartbeat(timeout=10)

            print(f"[Offboard v{self._vehicle_id()}] Heartbeat received.")

            self._connected = True

        except Exception as e:

            print(f"[Offboard] Connection failed: {e}")

    def _send_position_setpoint(self):
        """
        Stream a ``SET_POSITION_TARGET_LOCAL_NED`` message to PX4.

        Converts the stored ENU target position to the NED frame expected by
        MAVLink (north = +y_enu, east = +x_enu, down = −z_enu) and sends it
        with the ``_USE_POSITION`` typemask so only position and yaw are active.
        """

        if self._offboard_conn is None:
            return

        ex, ey, ez = self.target_pos

        # ENU → NED frame conversion
        north = ey
        east = ex
        down = -ez

        yaw_rad = np.deg2rad(self.target_yaw_deg)

        self._offboard_conn.mav.set_position_target_local_ned_send(
            self._timestamp_ms(),
            self._target_system(),
            1,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            _USE_POSITION,
            north,
            east,
            down,
            0.0, 0.0, 0.0,  # velocity (ignored)
            0.0, 0.0, 0.0,  # acceleration (ignored)
            yaw_rad,
            0.0,            # yaw rate (ignored)
        )

    def _set_mode_offboard(self):
        """
        Send a ``MAV_CMD_DO_SET_MODE`` command requesting PX4 OFFBOARD mode (custom mode 6).
        """

        self._offboard_conn.mav.command_long_send(
            self._target_system(),
            1,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6,      # PX4 custom mode: OFFBOARD
            0, 0, 0, 0, 0,
        )

    def _send_arm(self):
        """
        Send a ``MAV_CMD_COMPONENT_ARM_DISARM`` command to arm the vehicle.

        The magic number ``21196`` in param 7 is the PX4 force-arm confirmation
        code, required when arming outside of a safety-check-passing state.
        """

        self._offboard_conn.mav.command_long_send(
            self._target_system(),
            1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,      # 1 = arm, 0 = disarm
            0, 0, 0, 0, 0,
            21196,  # Force-arm confirmation code
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def set_target_position(self, position_enu):
        """
        Update the desired position while the controller is running.

        Args:
            position_enu: Array-like ``(x, y, z)`` in ENU metres.
        """
        self.target_pos = np.array(position_enu, dtype=float)

    def set_target_yaw_deg(self, yaw_deg):
        """
        Update the desired yaw while the controller is running.

        Args:
            yaw_deg (float): New target yaw in degrees, measured from East.
        """
        self.target_yaw_deg = float(yaw_deg)