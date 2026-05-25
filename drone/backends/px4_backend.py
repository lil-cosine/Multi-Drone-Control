from pegasus.simulator.logic.backends import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)

from drone.drone_utils.pegasus_backend import PegasusBackend


class PX4Backend(PegasusBackend):

    def __init__(
        self,
        pg,
        vehicle_id,
        px4_vehicle_model="gazebo-classic_iris",
    ):
        self.vehicle_id = vehicle_id

        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "px4_autolaunch": True,
            "px4_dir": pg.px4_path,
            "px4_vehicle_model": px4_vehicle_model,
            "enable_lockstep": True,
        })

        self._backend = PX4MavlinkBackend(mavlink_config)

    def start(self):
        self._backend.start()

    def stop(self):
        self._backend.stop()

    def update(self, dt):
        pass

    @property
    def backend(self):
        return self._backend