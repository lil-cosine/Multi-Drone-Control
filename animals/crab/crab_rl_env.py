import os

from isaaclab_tasks.direct.crab.crab_env import CrabEnvCfg

from animals.crab.crab_usd_cfg import get_crab_cfg
from utils import get_project_path


CRAB_TASK_NAME = "Isaac-Crab-Direct-v0"
CRAB_RESUME_PATH = os.path.join(get_project_path(), "assets", "animals", "crab", "crab_direct.pth")  # "/data/assets/animals/crab/crab_direct.pth"

def crab_env_cfg(init_pos=(6.0, 0.0, None), prim_path="/World/robot"):
    """Build a CrabEnvCfg with the given spawn position.

    Args:
        init_pos: (x, y, z) spawn position. Pass None for any axis to keep
                  the training-environment default for that axis.
        prim_path: USD prim path for this crab. Must be unique per instance.
    """
    cfg = CrabEnvCfg()
    cfg.robot = get_crab_cfg(init_pos=init_pos, prim_path=prim_path)
    cfg.action_scale_multiply = 100


    return cfg


def crab_task_cfg():
    return CRAB_TASK_NAME, CRAB_RESUME_PATH