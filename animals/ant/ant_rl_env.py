import os

from isaaclab_tasks.direct.ant.ant_env import AntEnvCfg

from animals.ant.ant_usd_cfg import get_ant_cfg
from utils import get_project_path


ANT_TASK_NAME = "Isaac-Ant-Direct-v0"
ANT_RESUME_PATH = os.path.join(
    get_project_path(), "assets", "animals", "ant_direct.pth"
)


def ant_env_cfg(init_pos=(6.0, 0.0, None), prim_path="/World/robot"):
    """Build an AntEnvCfg with the given spawn position.

    The AntEnvCfg inherits from DirectRLEnvCfg via LocomotionEnv and carries
    several reward-shaping fields (heading_weight, energy_cost_scale, etc.)
    that are left at their training defaults here. Override them after calling
    this function if you need different inference-time behaviour, e.g.:

        cfg = ant_env_cfg(init_pos=(8.0, 0.0, None), prim_path="/World/ant_0")
        cfg.heading_weight = 0.0   # disable heading reward at inference

    Args:
        init_pos: (x, y, z) spawn position. Pass None for z to keep the
                  asset's default ground-contact height.
        prim_path: USD prim path for this ant. Must be unique per instance.
    """
    cfg = AntEnvCfg()

    # AntEnvCfg.robot is an ArticulationCfg, same pattern as the crab.
    # get_ant_cfg deepcopies ANT_CFG and patches prim_path + init_state.pos.
    cfg.robot = get_ant_cfg(init_pos=init_pos, prim_path=prim_path)

    # Single-agent inference: one env, minimal spacing.
    # The training default is num_envs=4096 / env_spacing=4.0 — override so
    # we don't allocate a 4096-environment scene for a single deployed agent.
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 4.0
    cfg.action_scale_multiply = 30


    return cfg


def ant_task_cfg():
    return ANT_TASK_NAME, ANT_RESUME_PATH