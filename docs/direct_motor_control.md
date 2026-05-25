# Direct Motor Controller Guide

**Control mode:** `ControlMode.DIRECT`

---

## Overview

The `DirectMotorController` is the lowest-level control interface available. It bypasses all flight controller logic — there is no stabilisation, no attitude estimation, and no safety envelope. Your code is responsible for sending four motor thrust values every simulation step.

This makes it the right choice when:

- You are training or deploying a learned policy (e.g. reinforcement learning) that outputs raw motor commands
- You need deterministic, lag-free actuation with no PID or middleware interference
- You are doing research into low-level cooperative behaviours such as formation holding, payload transport, or collision avoidance at the motor level

It is the wrong choice when your algorithm produces high-level position or velocity targets — use the PX4 offboard controller for that instead.

---

## Motor Convention

Each drone has four motors indexed 0–3. The mixer in `hover_pid_controller.py` documents the layout and serves as a useful reference for understanding the sign conventions:

```
M0 (front-right):  throttle - pitch - roll
M1 (back-left):    throttle + pitch + roll
M2 (front-left):   throttle - pitch + roll
M3 (back-right):   throttle + pitch - roll
```

Motor values are floats passed directly to the backend. The physical range is **0.0 (off) to 1100.0 (full thrust)**. Values outside this range should be clipped before submission.

---

## Basic Setup

### 1. Attach the controller to a drone

```python
from direct_motor_controller import DirectMotorController
from control_mode import ControlMode

# Assuming `drone` is your drone instance
controller = DirectMotorController(drone)
controller.post_init()
```

### 2. Step the controller

Call `post_step(action)` once per simulation tick, passing a length-4 array of motor commands:

```python
import numpy as np

motors = np.array([500.0, 500.0, 500.0, 500.0])  # symmetric hover approximation
controller.post_step(motors)
```

Passing `None` is safe — the controller will send zeros (all motors off):

```python
controller.post_step(None)  # emergency stop
```

### 3. Reset between episodes

```python
controller.reset()  # clears nothing internally, but call it for interface consistency
```

---

## Multi-Drone Setup

For cooperative tasks, instantiate one controller per drone and step all of them inside your main loop:

```python
from direct_motor_controller import DirectMotorController
import numpy as np

NUM_DRONES = 4

controllers = [DirectMotorController(drone) for drone in drones]

for ctrl in controllers:
    ctrl.post_init()

# Main simulation loop
while not done:
    # actions: shape (NUM_DRONES, 4) — one motor command vector per drone
    actions = policy.step(observations)

    for i, ctrl in enumerate(controllers):
        ctrl.post_step(actions[i])

    observations = env.step()
```

Each controller is fully independent. There is no shared state between them — coordination is entirely the responsibility of your policy or algorithm.

---

## Algorithmic / Policy-Based Control

### Passing a learned policy's output

If you are using an RL policy or any algorithm that produces a vector of motor commands, the workflow is straightforward:

```python
import numpy as np

class MultiDroneDirectEnv:
    """
    Minimal wrapper showing how a learned policy feeds into
    DirectMotorController for each drone in the swarm.
    """

    def __init__(self, drones):
        self.controllers = [DirectMotorController(d) for d in drones]
        for ctrl in self.controllers:
            ctrl.post_init()

    def step(self, policy_actions: np.ndarray):
        """
        Args:
            policy_actions: np.ndarray of shape (n_drones, 4).
                            Values should be pre-clipped to [0, 1100].
        """
        assert policy_actions.shape == (len(self.controllers), 4)

        for i, ctrl in enumerate(self.controllers):
            ctrl.post_step(policy_actions[i])

    def reset(self):
        for ctrl in self.controllers:
            ctrl.reset()
```

### Action normalisation

Many RL frameworks output actions in `[-1, 1]` or `[0, 1]`. Rescale before passing to the controller:

```python
MOTOR_MIN = 0.0
MOTOR_MAX = 1100.0

def rescale_actions(normalised: np.ndarray) -> np.ndarray:
    """Map actions from [0, 1] to [MOTOR_MIN, MOTOR_MAX]."""
    return normalised * (MOTOR_MAX - MOTOR_MIN) + MOTOR_MIN
```

### Cooperative task example — distributed payload lift

A common cooperative task is having multiple drones share the load of a suspended payload. A decentralised policy might look like this:

```python
def cooperative_lift_step(controllers, observations, load_distribution):
    """
    Each drone adjusts its thrust based on its share of the total required lift.

    Args:
        controllers:       List of DirectMotorController instances.
        observations:      Per-drone state observations (position, attitude, etc.).
        load_distribution: 1-D array of weights summing to 1.0 — each drone's
                           fraction of the total required thrust.
    """
    BASE_HOVER = 450.0

    for i, (ctrl, obs, share) in enumerate(
        zip(controllers, observations, load_distribution)
    ):
        throttle = BASE_HOVER * share * len(controllers)

        # Attitude correction keeps each drone level independently
        roll_err  = -obs.roll
        pitch_err = -obs.pitch

        motors = np.array([
            throttle - pitch_err - roll_err,
            throttle + pitch_err + roll_err,
            throttle - pitch_err + roll_err,
            throttle + pitch_err - roll_err,
        ])

        motors = np.clip(motors, 0.0, 1100.0)
        ctrl.post_step(motors)
```

---

## Gotchas

**No stabilisation.** If your policy produces a bad action even for one step, the drone will respond immediately. There is no inner PID loop to absorb transient errors.

**Motor symmetry matters.** For a hover, all four motors should produce equal thrust. Any imbalance causes rotation. If your drones are drifting or spinning unexpectedly, check that your action space is centred correctly.

**Clipping is your responsibility.** The controller passes values to the backend as-is. Always clip to `[0.0, 1100.0]` before calling `post_step`.

**Reset does not stop motors.** Calling `reset()` does not send a zero command. If you need to stop all motors on reset, do it explicitly:

```python
ctrl.post_step(np.zeros(4))
ctrl.reset()
```