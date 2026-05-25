# PX4 Offboard Controller Guide

**Control mode:** `ControlMode.PX4_OFFBOARD`

---

## Overview

The `OffboardController` drives a drone to a 3-D position target by sending MAVLink `SET_POSITION_TARGET_LOCAL_NED` messages to PX4 over a UDP link. PX4 handles low-level stabilisation internally — your algorithm only needs to produce **where** each drone should be, not how to get there.

This makes it the right choice when:

- Your cooperative algorithm reasons in terms of positions (formations, coverage paths, task allocation)
- You want PX4's built-in safety features (geofencing, failsafe, motor mixing) to remain active
- You are running hardware-in-the-loop (HITL) or real vehicles where bypassing the flight controller is unacceptable
- Your algorithm outputs waypoints or continuous position/yaw references on a per-drone basis

It is a heavier setup than `DirectMotorController` because it requires a live MAVLink connection and must work through a startup sequence before flight begins.

---

## Coordinate Frames

The controller accepts positions in **ENU (East-North-Up)** and converts internally to the NED frame PX4 expects:

| ENU axis | NED axis |
|----------|----------|
| +X (East)  | North → mapped from +Y ENU |
| +Y (North) | East  → mapped from +X ENU |
| +Z (Up)    | Down  → negated Z ENU      |

You do not need to think about NED in your algorithm. Always supply ENU coordinates.

**Yaw convention:** degrees measured clockwise from East (ENU convention). 0° points East, 90° points North.

---

## Network Topology

Each drone listens for MAVLink on a unique UDP port:

```
port = 14580 + vehicle_id
```

For a four-drone swarm with vehicle IDs 0–3, the ports are 14580, 14581, 14582, 14583. The controller opens a `udpout` connection to `127.0.0.1:<port>` using its own source system ID of `200 + vehicle_id` to avoid MAVLink ID collisions.

---

## Startup Sequence

PX4 will not accept an OFFBOARD mode switch unless setpoints are already being streamed. The controller automates this with a five-stage state machine:

```
[wait 500 steps] → connect → [prime 200 steps] → mode switch → [arm 100 steps] → active
```

| Stage | What happens |
|-------|-------------|
| **wait** | Idle — gives the simulator time to fully initialise |
| **connect** | Opens the UDP link and waits for a MAVLink heartbeat |
| **prime** | Streams position setpoints so PX4 sees an active offboard stream |
| **mode** | Sends `MAV_CMD_DO_SET_MODE` to switch PX4 into OFFBOARD (custom mode 6) |
| **arm** | Sends `MAV_CMD_COMPONENT_ARM_DISARM` with force-arm confirmation |
| **active** | Continuously streams the target setpoint — your algorithm drives from here |

**Time to active (at 50 Hz):** approximately 24 seconds. Plan your task execution to start after all drones have reached the `active` state.

---

## Basic Setup

### 1. Attach the controller to a drone

```python
from offboard_controller import OffboardController

controller = OffboardController(
    drone,
    target_pos=(0.0, 0.0, 1.5),   # ENU: hover at 1.5 m
    target_yaw_deg=0.0,
)
controller.post_init()
```

### 2. Step the controller

Call `post_step()` once per simulation tick. The `action` argument is unused — pass `None` or omit it:

```python
controller.post_step()
```

### 3. Update the target during flight

Once the state machine has reached `active`, update the target at any time:

```python
controller.set_target_position([2.0, 3.0, 1.5])   # move to new ENU position
controller.set_target_yaw_deg(90.0)                # face North
```

### 4. Reset between episodes

```python
controller.reset()
# Note: the MAVLink connection is preserved — no reconnection overhead
```

---

## Multi-Drone Setup

Instantiate one controller per drone. Each gets its own UDP port automatically via `vehicle_id`:

```python
from offboard_controller import OffboardController

NUM_DRONES = 4

# Assign staggered starting positions so drones don't overlap on takeoff
start_positions = [
    (0.0, 0.0, 1.5),
    (2.0, 0.0, 1.5),
    (0.0, 2.0, 1.5),
    (2.0, 2.0, 1.5),
]

controllers = [
    OffboardController(drone, target_pos=pos)
    for drone, pos in zip(drones, start_positions)
]

for ctrl in controllers:
    ctrl.post_init()

# Main simulation loop
while not done:
    for ctrl in controllers:
        ctrl.post_step()

    observations = env.step()
```

---

## Algorithmic / Policy-Based Control

### Checking readiness before issuing targets

Because the startup sequence takes time, your algorithm should gate task execution on all drones being active:

```python
def all_active(controllers):
    return all(c._state == c._STATE_ACTIVE for c in controllers)

# Warmup loop
while not all_active(controllers):
    for ctrl in controllers:
        ctrl.post_step()
    env.step()

print("All drones active — beginning cooperative task")
```

### Issuing waypoints from a planner

Once active, a high-level planner can push new targets as often as needed:

```python
def execute_formation(controllers, formation_positions):
    """
    Move all drones to the given ENU positions simultaneously.

    Args:
        controllers:        List of OffboardController instances.
        formation_positions: List of (x, y, z) ENU tuples, one per drone.
    """
    for ctrl, pos in zip(controllers, formation_positions):
        ctrl.set_target_position(pos)
```

### Cooperative task example — dynamic formation flight

This example shows a centralised coordinator updating drone positions every step to maintain a rotating formation:

```python
import numpy as np

def rotating_formation_step(controllers, center, radius, angle_rad, altitude):
    """
    Arrange drones in an evenly spaced ring around `center`, rotating over time.

    Args:
        controllers: List of OffboardController instances.
        center:      (x, y) ENU coordinates of the formation centre.
        radius:      Radius of the ring in metres.
        angle_rad:   Current rotation angle in radians (increment each step).
        altitude:    Target altitude in metres.
    """
    n = len(controllers)

    for i, ctrl in enumerate(controllers):
        offset_angle = angle_rad + (2 * np.pi * i / n)

        x = center[0] + radius * np.cos(offset_angle)
        y = center[1] + radius * np.sin(offset_angle)

        # Each drone faces the centre of the formation
        yaw_to_centre = np.degrees(
            np.arctan2(center[1] - y, center[0] - x)
        )

        ctrl.set_target_position([x, y, altitude])
        ctrl.set_target_yaw_deg(yaw_to_centre)
```

```python
# Caller — increment angle each step for continuous rotation
angle = 0.0
ANGULAR_VELOCITY = 0.01  # radians per step

while task_running:
    for ctrl in controllers:
        ctrl.post_step()

    rotating_formation_step(
        controllers,
        center=(5.0, 5.0),
        radius=3.0,
        angle_rad=angle,
        altitude=2.0,
    )

    angle += ANGULAR_VELOCITY
    env.step()
```

### Cooperative task example — task allocation with position targets

A decentralised allocation algorithm (e.g. auction-based or Hungarian assignment) can feed directly into the controller's public API:

```python
def assign_and_fly(controllers, task_positions, drone_positions):
    """
    Solve a simple nearest-drone assignment and dispatch each drone to its task.

    Args:
        controllers:     List of OffboardController instances.
        task_positions:  List of (x, y, z) ENU target positions.
        drone_positions: Current ENU positions of each drone.
    """
    from scipy.optimize import linear_sum_assignment

    n = len(controllers)

    # Build cost matrix: Euclidean distance from each drone to each task
    cost = np.array([
        [np.linalg.norm(np.array(dp) - np.array(tp))
         for tp in task_positions]
        for dp in drone_positions
    ])

    drone_idx, task_idx = linear_sum_assignment(cost)

    for d, t in zip(drone_idx, task_idx):
        controllers[d].set_target_position(task_positions[t])
```

---

## Gotchas

**Startup takes time.** With default constants at 50 Hz, the full sequence (wait → active) takes roughly 24 seconds. For training loops, either reduce `_PRIME_START_STEPS` or account for the warmup in your episode length.

**MAVLink connection survives reset.** `reset()` rewinds the state machine to `wait` but does not close the UDP socket. On the next episode the controller will reconnect faster (it skips the heartbeat wait) but will still step through the full state sequence before becoming active again. If you want to avoid this, reduce `_PRIME_START_STEPS` to a small number after the first episode.

**Setpoints must keep streaming.** PX4 will exit OFFBOARD mode if setpoints stop arriving for more than ~0.5 seconds. Always call `post_step()` every simulation tick even if your target has not changed.

**One controller per vehicle ID.** If you run multiple instances pointing at the same `vehicle_id`, they will share a port and their MAVLink messages will collide. Each drone must have a unique `vehicle_id` in its backend.

**ENU not NED.** All public-facing methods (`set_target_position`, the constructor's `target_pos`) use ENU. Do not pre-convert to NED — the controller does this internally.