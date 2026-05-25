# Multi-Drone Control
This project is a template for programatic control of multiple drones through Isaac Sim. It supports both drone control through Mavlink commands over PX4 and QGroundControl as well as direct commands to the individual motors. This project also supports the addition of other robots controlled by policies trained through Isaac Lab.

- **Demo Video:** 

https://github.com/user-attachments/assets/5acbb953-dfd2-4f62-a566-0aee957c9afb


*Demo: Two crabs move on the left. The left two drone are controlled by a custom PID controller directly commanding the motors. The center two drones do not move. The right two drones are controlled by Mavlink commands.*

## Drone Simulation Setup Guide

This guide will help you set up and run the drone control demo using Isaac Sim, Pegasus, and QGroundControl.

---

### Prerequisites

- **Git LFS:**  
  Install [Git LFS](https://git-lfs.com/) to handle large `.usd` files.

- **Isaac Sim & Isaac Lab:**  
  Follow the [Isaac Sim installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html) to install Isaac Sim and set up the IsaacLab conda/venv environment.

---

### Installation Steps

1. **Clone IsaacLab and Install RL Games:**
    ```bash
    git clone https://github.com/lil-cosine/IsaacLab.git -b crab-rl
    cd IsaacLab
    ./isaaclab.sh --install rl_games
    ```

2. **Install PX4 Autopilot (for Pegasus):**  
   Refer to the [Pegasus PX4 Installation Guide](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html#installing-px4-autopilot).

   **For Ubuntu 24.04:**  
   Before installing PX4, run:
    ```bash
    sudo apt update
    sudo apt install gcc-12 g++-12
    which gcc-12  # Verify installation
    export CC=/usr/bin/gcc-12
    export CXX=/usr/bin/g++-12
    ```

3. **Clone and Install Pegasus (v5.1.0):**
    > **Note:** Clone Pegasus **outside** the project directory.
    ```bash
    git clone https://github.com/PegasusSimulator/PegasusSimulator.git
    cd PegasusSimulator
    git checkout v5.1.0
    pip install -e extensions/pegasus.simulator
    ```
    Once Pegasus is installed, add the path of the PX4 directory to the Pegasus config file found at `PegasusSimulator/extensions/pegasus.simulator/config/configs.yaml`

4. **Download and Set Up QGroundControl:**  
   Download QGroundControl from [here](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html).  
   You will use this software to control the drone (joystick recommended).
   > **Note:** If you encounter `Missing Parameter` errors, download a daily build from [here](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/releases/daily_builds.html)

---

### Creating a Controller with PX4
* Details [here](docs/px4_controller.md)

### Creating a Direct Motor Controller
* Details [here](docs/direct_motor_control.md)

### Creating and Training New Animal Environment
* Details [here](docs/training_rl.md)

---

### Running the Drone Teleop Demo

1. **Start QGroundControl:**
    - Open QGroundControl.
    - Go to `Vehicle Configuration` and enable the joystick controller.

2. **Run the Drone Demo Script:**
    - With your IsaacLab environment activated, run:
      ```bash
      python3 standalone/drone_main.py
      ```
      If you encounter a "module not found" error, try:
      ```bash
      python -m standalone.drone_main
      ```

3. **Observe the Drones:**
    - Once Isaac Sim launches, switch back to QGroundControl to ensure it has connected to Pegasus Sim.
    - The two drones on the left (direct motor control) should begin flying in the Isaac Sim window.
    - After the lauch sequence, the two drones on the right should begin flying to their designated hold positions. 
        > **Note:** It is normal for Isaac Sim to freeze during the PX4 launch sequence. Everything should resume as normal after a few seconds. 

---

**Tip:**  
For best results, keep IsaacLab, Pegasus, and QGroundControl in separate directories outside this project folder.
