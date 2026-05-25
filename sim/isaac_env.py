from typing import Any, SupportsFloat
import os

import gymnasium
from gymnasium.core import ObsType, ActType
from isaaclab.sim import GroundPlaneCfg
from isaaclab.sim import DomeLightCfg
import isaacsim.core.utils.torch as torch_utils

from utils import get_project_path

SIM_DT = 1 / 250

class IsaacEnv(gymnasium.Env):
    def __init__(self, layout="air"):
        from isaacsim import SimulationApp
        from sim.app import get_app
        self.app: SimulationApp = get_app()

        from isaaclab.sim import SimulationCfg
        from sim.fake_world import FakeWorld

        self.sim_cfg = SimulationCfg(dt=SIM_DT, render_interval=1, device="cpu")
        self.sim_cfg.use_fabric = False
        self.world: FakeWorld = FakeWorld(self.sim_cfg, physics_dt=SIM_DT)
        self.sim = self.world
        self.load_layout(layout)
        self.init_reset = False
        self.seed(42)
        self.dt = SIM_DT
        
        self._components_added = False

    def get_world_settings(self):
        return {}

    def load_layout(self, layout):
        import isaaclab.sim as sim_utils

        if layout == "air":
            cfg = GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="average",
                    restitution_combine_mode="average",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            )
            cfg.func("/World/defaultGroundPlane", cfg)

            cfg = DomeLightCfg(
                color=(0.1, 0.1, 0.1),
                enable_color_temperature=True,
                color_temperature=5500,
                intensity=10000,
            )
            cfg.func(prim_path="/World/defaultDomeLight", cfg=cfg, translation=(0.0, 0.0, 10.0))

        elif layout == "water":
            raise NotImplementedError("Water layout not implemented yet")
        #     cfg = sim_utils.UsdFileCfg(
        #         usd_path= os.path.join(get_project_path(), "assets/worlds/grid_with_stand_water.usd"),
        #     )
        #     cfg.func("/World/layout", cfg)
        elif layout  == "grid":
            cfg = sim_utils.UsdFileCfg(
                usd_path= os.path.join(get_project_path(), "assets/worlds/grid_with_stand.usd"),
            )
            cfg.func("/World/layout", cfg)
        else:
            raise ValueError(f"Unknown layout type {layout}")

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.seed(seed)
        
        if not self.init_reset:
            self.world.reset(soft=False)
            self.post_init()
            self._components_added = True
            self.init_reset = True
        else:
            self.world.reset(soft=True)
        
        return {}, {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.world.step(render=True)
        
        return {}, 0.0, False, False, {}

    def post_init(self):
        """Override this method to add components after world initialization."""
        pass

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def __del__(self):
        self.close()

    def close(self):
        """Clean up simulation resources."""
        if hasattr(self, 'sim') and self.sim is not None:
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()