import torch
from isaaclab_tasks.direct.crab.crab_env import CrabEnvCfg

from animals.crab.crab_rl_env import crab_env_cfg, crab_task_cfg
from animals.rl_agent_controller import RlAnimalController
from sim.isaac_env import IsaacEnv


class PretrainedRlAnimalController(RlAnimalController):
    def __init__(
        self,
        parent_env: IsaacEnv,
        init_pos: tuple = None,
        prim_path: str = "/World/robot",
        env_cfg: CrabEnvCfg = None,
        task_cfg: tuple = None,
    ):
        if env_cfg is None:
            env_cfg = crab_env_cfg(
                init_pos=init_pos if init_pos is not None else (6.0, 0.0, None),
                prim_path=prim_path,
            )
        if task_cfg is None:
            task_cfg = crab_task_cfg()

        env_cfg.sim.device = parent_env.sim.device
        super().__init__(parent_env, env_cfg)
        self.env_cfg = env_cfg
        from animals.agent import Agent

        self.agent = Agent(self, task_cfg)

        self.current_step = 0
        self.last_obs = None

    def pre_decimation(self):
        obs = self.last_obs
        action = self.agent.get_action(obs)
        self.actions = action.clone()

    def pre_step(self):
        if self.current_step % self.env_cfg.decimation == 0:
            self.pre_decimation()
        self.apply_action()

    def post_step(self):
        super().post_step()

        self.current_step += 1
        if self.current_step % self.env_cfg.decimation == 0:
            self.post_decimation()

    def post_decimation(self):
        self.episode_length_buf += 1 

        self.reset_terminated, self.reset_time_outs = self.get_dones()
        self.reset_buf = self.reset_terminated or self.reset_time_outs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        self.update_obs()

    def reset(self):
        super().reset()
        self.agent.init(self.last_obs)