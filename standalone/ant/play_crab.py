from animals.pretrained_rl_agent_controller import PretrainedRlAnimalController
from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from utils import add_gamepad_callback, set_active_camera
from omni.isaac.lab_tasks.direct.crab.crab_env import CrabEnvCfg
from animals.crab.crab_usd_cfg import get_crab_cfg
from drone.manipulators import ManipulatorState
from sim.isaac_env import IsaacEnv


def default_rl_env(pos, prim_path):
    cfg = CrabEnvCfg()
    cfg.robot = get_crab_cfg(prim_path=prim_path, init_pos=pos)  # Keep the z-position same as training environment

    return cfg


class CrabEnv(IsaacEnv):
    def __init__(self, cfg: CrabEnvCfg):
        super().__init__(layout="grid")

        self.sim = self.world

        self.controllers = []

        positions = [
            (6.0, 0.0, None),
            (8.0, 0.0, None),
        ]

        for i, pos in enumerate(positions):
            prim_path = f'/World/crab_{i}'
            controller = PretrainedRlAnimalController(parent_env=self, env_cfg=default_rl_env(pos, prim_path))
            self.controllers.append(self.controller)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        for controller in self.controllers:
            controller.reset()

    def step(self, _):
        for controller in self.controllers:
            controller.pre_step()
        super().step(None)
        for controller in self.controllers:
            controller.post_step()

    def post_init(self):
        for controller in self.controllers:
            controller.post_init()


env_cfg = CrabEnvCfg()
env = CrabEnv(env_cfg)

# create drone controller
# drone_controller = DroneControllerQGroundControl(env)

mani_state = ManipulatorState()
add_gamepad_callback(mani_state.gamepad_callback)

set_active_camera("/World/drone/arm/arm_base/Camera")
env.reset()
# drone_controller.post_init()
while app.is_running():

    # drone_controller.post_step(mani_state.as_action())
    for _ in range(4):
        env.step(None)

    env.sim.render()
    # obs = env.crab_controller.last_obs

env.close()
app.close()
