# from enum import Enum


# class ControlMode(Enum):
#     DIRECT = "direct"
#     PX4_OFFBOARD = "px4_offboard"
#     PX4_QGC = "px4_qgc"

from enum import Enum


class ControlMode(Enum):
    """
    Enum representing the available control modes for the drone.

    Modes:
        DIRECT:        Direct motor command control, bypassing any flight controller.
        PX4_OFFBOARD:  PX4 offboard control via MAVLink (programmatic position/velocity setpoints).
        PX4_QGC:       PX4 control via QGroundControl (manual or mission-based).
    """

    DIRECT = "direct"
    PX4_OFFBOARD = "px4_offboard"
    PX4_QGC = "px4_qgc"