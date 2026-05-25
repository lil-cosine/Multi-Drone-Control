from typing import Any, List

from gymnasium.core import ObsType

from animals.pretrained_rl_agent_controller import PretrainedRlAnimalController
from drone.drone_controller import DroneController
from sim.isaac_env import IsaacEnv


class DroneEnv(IsaacEnv):
    """
    Multi-agent drone + crab environment.

    Callers supply factory callables so that each controller is constructed
    after IsaacEnv.__init__ has set up the world:

        drone_controller_factories = [
            lambda env: DroneControllerQGroundControl(env),
            lambda env: DroneControllerQGroundControl(env),
        ]
        crab_controller_factories = [
            lambda env: PretrainedRlAnimalController(env),
            lambda env: PretrainedRlAnimalController(env),
        ]

    If no factories are provided the env falls back to one default drone and
    one default crab controller (preserving the original behaviour).
    """

    def __init__(
        self,
        drone_controller_factories=None,
        crab_controller_factories=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if drone_controller_factories is None:
            drone_controller_factories = [lambda env: DroneController(parent_env=env)]
        if crab_controller_factories is None:
            crab_controller_factories = [lambda env: PretrainedRlAnimalController(parent_env=env)]

        self.drone_controllers: List[DroneController] = [
            factory(self) for factory in drone_controller_factories
        ]
        self.crab_controllers: List[PretrainedRlAnimalController] = [
            factory(self) for factory in crab_controller_factories
        ]

        self.drone_controller = self.drone_controllers[0] if self.drone_controllers else None

    def step(self, action):
        for crab in self.crab_controllers:
            crab.pre_step()

        super().step(action)

        for drone in self.drone_controllers:
            drone.post_step(action)

        for crab in self.crab_controllers:
            crab.post_step()

        return {}, 0.0, False, False, {}

    def post_init(self):
        for drone in self.drone_controllers:
            drone.post_init()
        for crab in self.crab_controllers:
            crab.post_init()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        for drone in self.drone_controllers:
            drone.reset()
        for crab in self.crab_controllers:
            crab.reset()
        return {}, {}

    def close(self):
        for drone in self.drone_controllers:
            drone.close()