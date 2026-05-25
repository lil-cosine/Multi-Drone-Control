import numpy as np
from pegasus.simulator.logic.backends import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from drone.drone_controller import DroneController


class DroneControllerQGroundControl(DroneController):
    """
    DroneController that connects to QGroundControl via PX4/Mavlink.

    Parameters
    ----------
    parent_env : IsaacEnv
        The parent simulation environment.
    vehicle_id : int
        Unique vehicle ID for this drone. Each drone in the same simulation
        must use a different ID (0, 1, 2, ...). The ID also determines the
        Mavlink UDP port offsets so QGroundControl can distinguish vehicles.
    stage_prefix : str or None
        USD stage path for this drone. Defaults to "/World/drone_<vehicle_id>".
    init_pos : array-like or None
        World-frame spawn position [x, y, z]. Defaults to DroneController default.
    init_rot : array-like or None
        Euler XYZ spawn rotation in degrees [rx, ry, rz]. Defaults to DroneController default.
    """

    def __init__(
        self,
        parent_env,
        vehicle_id: int = 0,
        stage_prefix: str = None,
        init_pos=None,
        init_rot=None,
    ):
        self._vehicle_id = vehicle_id
        resolved_prefix = stage_prefix if stage_prefix is not None else f"/World/drone_{vehicle_id}"
        super().__init__(
            parent_env,
            stage_prefix=resolved_prefix,
            init_pos=init_pos,
            init_rot=init_rot,
        )

    def get_backend(self):
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": self._vehicle_id,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": "gazebo-classic_iris",
            "enable_lockstep": True,
        })
        return PX4MavlinkBackend(mavlink_config)

    def close(self):
        if hasattr(self, "backend") and self.backend:
            print(f"[DroneControllerQGC id={self._vehicle_id}] Stopping PX4 backend...")
            self.backend.stop()