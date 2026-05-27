"""
Teleoperation script with modular drone backends/controllers.

Overview
--------
This script initialises a mixed-fleet simulation environment containing drones
running three different control modes and two pretrained RL-driven crab agents.
It then runs a real-time teleoperation loop driven by gamepad input until the
exit button is pressed, and saves the recorded commands to disk.

Fleet composition (defined in DRONE_CONFIGS)
--------------------------------------------
- Vehicles 0–1 : PX4 offboard control  — position setpoints via MAVLink UDP
- Vehicles 2–3 : Hover PID control     — altitude + attitude hold via direct motor commands
- Vehicles 4–5 : Direct motor control  — raw 4-motor commands, no stabilisation

Control modes
-------------
See drone/control/ for the controller implementations and the accompanying
docs for guidance on choosing between modes for cooperative tasks:
  - direct_motor_controller_guide.md
  - px4_offboard_controller_guide.md

Gamepad
-------
The exit button (MENU2) ends the loop. Additional gamepad callbacks can be
registered via add_gamepad_callback() before the loop starts.

Output
------
Motor commands are accumulated in `commands` and saved to
recordings/output/episode_actions.pkl on exit.
"""

import numpy as np

from sim.app import init_app
init_app()

from carb.input import GamepadInput

from drone.drone_controller import DroneController

from drone.backends.direct_backend import DirectBackend
from drone.backends.px4_backend import PX4Backend

from drone.control.direct_motor_controller import DirectMotorController
from drone.control.offboard_controller import OffboardController

from animals.pretrained_rl_agent_controller import (
    PretrainedRlAnimalController,
)

from drone.control.hover_pid_controller import (
    HoverPIDController,
)

from drone.teleop.drone_teleop_env import DroneTeleOpEnv

from pegasus.simulator.logic.interface.pegasus_interface import (
    PegasusInterface,
)

from utils import (
    add_gamepad_callback,
    set_active_camera,
    GamepadButtonPressWatcher,
    save_pickle,
)

from animals.crab.crab_rl_env import crab_env_cfg, crab_task_cfg
from animals.ant.ant_rl_env import ant_env_cfg, ant_task_cfg

# ============================================================
# DRONE CONFIG
# ============================================================
#
# Each entry is a dict that fully describes one drone. The factory function
# below reads these dicts and wires up the correct backend + controller.
#
# Common fields
# -------------
# vehicle_id (int)  : Unique integer ID. Used as the USD stage index
#                     (/World/drone_<vehicle_id>) and, for PX4 drones, to
#                     derive the MAVLink UDP port (14580 + vehicle_id).
# mode (str)        : "px4_offboard" | "hover_pid" | "direct"
# init_pos (tuple)  : Spawn position in world-space ENU metres (x, y, z).
# init_rot (tuple)  : Spawn orientation as Euler angles in degrees (r, p, y).
#
# Mode-specific fields
# --------------------
# px4_offboard:
#   offboard_pos (tuple) : Initial ENU position target sent to PX4 on startup.
#
# hover_pid:
#   target_z (float)     : Desired hover altitude in metres.
#
# direct:
#   (no extra fields — motor commands must be supplied externally each step)
#
# To add a drone, append a new dict to the list. Ensure vehicle_id is unique
# and that init_pos does not overlap with an existing drone's spawn point.

DRONE_CONFIGS = [

    # --------------------------------------------------------
    # PX4 OFFBOARD DRONE
    # --------------------------------------------------------

    {
        "vehicle_id": 0,
        "mode": "px4_offboard",

        "init_pos": (0.0, 0.0, 0.0),
        "init_rot": (0.0, 0.0, 0.0),

        "offboard_pos": (0.0, 0.0, 1.0),    # hover at 1 m
    },

    {
        "vehicle_id": 1,
        "mode": "px4_offboard",

        "init_pos": (0.0, 2.0, 0.0),
        "init_rot": (0.0, 0.0, 0.0),

        "offboard_pos": (0.0, 0.0, 2.0),    # hover at 2 m
    },

    # --------------------------------------------------------
    # PID DIRECT MOTOR CONTROL DRONE
    # --------------------------------------------------------
    {
        "vehicle_id": 2,
        "mode": "hover_pid",

        "init_pos": (4.0, 0.0, 0.0),
        "init_rot": (0.0, 0.0, 0.0),

        "target_z": 1.0,
    },
    {
        "vehicle_id": 3,
        "mode": "hover_pid",

        "init_pos": (4.0, 2.0, 0.0),
        "init_rot": (0.0, 0.0, 0.0),

        "target_z": 2.0,
    },
    # --------------------------------------------------------
    # DIRECT MOTOR DRONE
    # --------------------------------------------------------
    {
        "vehicle_id": 4,
        "mode": "direct",

        "init_pos": (2.0, 0.0, 0.0),
        "init_rot": (0.0, 0.0, 0.0),
    },
    {
        "vehicle_id": 5,
        "mode": "direct",

        "init_pos": (2.0, 2.0, 0.0),
        "init_rot": (0.0, 0.0, 0.0),
    },
]


# ============================================================
# ANIMAL CONFIG
# ============================================================
#
# Each entry spawns one pretrained RL animal agent.
#
# Fields
# ------
# init_pos (tuple)  : Spawn position (x, y, None). The z coordinate is
#                     managed by the agent itself (ground-contact dependent).
# prim_path (str)   : USD prim path where the crab asset is loaded.

ANIMAL_CONFIGS = [
    {
        "type": "crab",
        "init_pos": (6.0, 0.0, None),
        "prim_path": "/World/robot_0",
    },
    {
        "type": "crab",
        "init_pos": (6.0, 2.0, None),
        "prim_path": "/World/robot_1",
    },
    {
        "type": "ant",
        "init_pos": (8.0, 0.0, None),
        "prim_path": "/World/robot_2",
    },
    {
        "type": "ant",
        "init_pos": (8.0, 2.0, None),
        "prim_path": "/World/robot_3",
    },
]


# ============================================================
# DRONE FACTORY
# ============================================================

def make_drone_factory(cfg):
    """
    Return a factory function that constructs and wires a DroneController for
    the given config dict when called with an environment instance.

    The factory pattern is used so that drone construction is deferred until
    the simulation environment exists. Each factory closes over its own `cfg`
    dict so all factories in the list are independent.

    Args:
        cfg (dict): A drone config dict from DRONE_CONFIGS.

    Returns:
        Callable[[env], DroneController]: A factory that accepts the
        environment and returns a fully configured DroneController.

    Raises:
        ValueError: If ``cfg["mode"]`` is not a recognised control mode.
    """

    def factory(env):

        vehicle_id = cfg["vehicle_id"]

        # --------------------------------------------------------
        # PX4 OFFBOARD
        # --------------------------------------------------------
        # Uses PX4Backend, which connects to a PX4 SITL (or hardware)
        # instance via PegasusInterface. The OffboardController drives
        # the drone to a fixed ENU position through the MAVLink startup
        # sequence (wait → connect → prime → mode → arm → active).

        if cfg["mode"] == "px4_offboard":

            backend = PX4Backend(
                PegasusInterface(),
                vehicle_id=vehicle_id,
            )

            drone = DroneController(
                env,
                backend=backend,
                controller=None,            # controller injected below
                stage_prefix=f"/World/drone_{vehicle_id}",
                init_pos=cfg["init_pos"],
                init_rot=cfg["init_rot"],
            )

            controller = OffboardController(
                drone,
                target_pos=cfg["offboard_pos"],
            )

            drone.set_controller(controller)

            return drone

        # --------------------------------------------------------
        # DIRECT MOTOR
        # --------------------------------------------------------
        # Uses DirectBackend, which bypasses PX4 entirely. The
        # DirectMotorController forwards raw 4-motor commands straight
        # to the simulator. No stabilisation is applied — the caller
        # must supply sensible motor values each step.

        elif cfg["mode"] == "direct":

            backend = DirectBackend()

            drone = DroneController(
                env,
                backend=backend,
                controller=None,
                stage_prefix=f"/World/drone_{vehicle_id}",
                init_pos=cfg["init_pos"],
                init_rot=cfg["init_rot"],
            )

            controller = DirectMotorController(drone)

            drone.set_controller(controller)

            return drone

        # --------------------------------------------------------
        # HOVER PID
        # --------------------------------------------------------
        # Also uses DirectBackend (no PX4), but wraps the motor
        # commands in a dual-loop PID that holds altitude and keeps
        # roll/pitch at zero. target_z sets the desired hover height.

        elif cfg["mode"] == "hover_pid":

            backend = DirectBackend()

            drone = DroneController(
                env,
                backend=backend,
                controller=None,
                stage_prefix=f"/World/drone_{vehicle_id}",
                init_pos=cfg["init_pos"],
                init_rot=cfg["init_rot"],
            )

            controller = HoverPIDController(
                drone,
                target_z=cfg.get("target_z", 1.0),
            )

            drone.set_controller(controller)

            return drone

        else:

            raise ValueError(f"Unknown mode: {cfg['mode']}")

    return factory


# ============================================================
# CRAB FACTORY
# ============================================================

ANIMAL_REGISTRY = {
    "crab": (crab_env_cfg, crab_task_cfg),
    "ant": (ant_env_cfg, ant_task_cfg),
}

def make_animal_factory(cfg):
    """
    Return a factory function that constructs a PretrainedRlAnimalController
    for the given crab config dict.

    Args:
        cfg (dict): A crab config dict from CRAB_CONFIGS.

    Returns:
        Callable[[env], PretrainedRlAnimalController]
    """

    animal_type = cfg["type"]
    if animal_type not in ANIMAL_REGISTRY:
        raise ValueError(f"Unknown animal type: '{animal_type}'. "
                         f"Registered types: {list(ANIMAL_REGISTRY)}")

    env_cfg_fn, task_cfg_fn = ANIMAL_REGISTRY[animal_type]

    def factory(env):
        return PretrainedRlAnimalController(
            env,
            init_pos=cfg["init_pos"],
            prim_path=cfg["prim_path"],
            env_cfg=env_cfg_fn(
                init_pos=cfg["init_pos"],
                prim_path=cfg["prim_path"],
            ),
            task_cfg=task_cfg_fn(),
        )

    return factory


# Build the factory lists passed to the environment. Order matches the index
# order used internally by DroneTeleOpEnv for step() and reset() calls.
drone_factories = [
    make_drone_factory(cfg)
    for cfg in DRONE_CONFIGS
]

animal_factories = [
    make_animal_factory(cfg)
    for cfg in ANIMAL_CONFIGS
]


# ============================================================
# ENVIRONMENT
# ============================================================
#
# DroneTeleOpEnv constructs all agents by calling their factories, lays out
# the scene according to `layout`, and exposes the standard step/reset/close
# interface used below.

env = DroneTeleOpEnv(
    layout="grid",
    drone_controller_factories=drone_factories,
    animal_controller_factories=animal_factories,
)


# ============================================================
# GAMEPAD
# ============================================================
#
# exit_watcher monitors the MENU2 button. When pressed, it sets
# exit_watcher.pressed = True, which terminates the main loop.
# Additional gamepad callbacks (e.g. for manual drone control) can be
# registered here with add_gamepad_callback() before the loop starts.

exit_watcher = GamepadButtonPressWatcher(GamepadInput.MENU2)

add_gamepad_callback(exit_watcher.gamepad_callback)


# ============================================================
# RUN
# ============================================================

commands = []  # Accumulates motor command history for post-episode analysis

# Point the viewport camera at the first drone in the config list.
set_active_camera(
    f"/World/drone_{DRONE_CONFIGS[0]['vehicle_id']}"
)

env.reset()

count = 0

while not exit_watcher.pressed:

    # Step all drone controllers and crab agents.
    # The zero action here is a placeholder; in a teleoperated or algorithmic
    # session you would replace np.zeros(4) with the active drone's command.
    env.step(np.zeros(4))

    # Render every 4 physics steps to reduce GPU load while keeping the
    # viewport responsive (~12–15 fps at a 50 Hz physics rate).
    if count % 4 == 0:
        env.sim.render()

    count += 1

# Persist the recorded command history so episodes can be replayed or used
# as demonstration data for imitation learning pipelines.
save_pickle(
    "recordings/output/episode_actions.pkl",
    commands,
)

print("Commands saved to episode_actions.pkl")

env.close()