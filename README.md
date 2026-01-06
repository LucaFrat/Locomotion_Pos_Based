# Locomotion Position Based (Isaac Lab)

This repository implements **End-to-End Position-Based Locomotion** using [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab).

It serves as a reproduction and extension of the concepts presented in the paper:
> **Advanced Skills by Learning Locomotion and Local Navigation End-to-End** > *Nikita Rudin, David Hoeller, Marko Bjelonic, and Marco Hutter* > [arXiv:2209.12827](https://arxiv.org/abs/2209.12827)

Unlike standard locomotion tasks that track a commanded velocity ($v_x, v_y, \omega_z$), this project trains the robot to reach a specific **target position** $(x_{goal}, y_{goal})$ in the world, giving the policy the freedom to choose its own path and velocity profile.

## Key Features

### 1. Position-Based Command Generator
Instead of velocity commands, the robot receives a 3D goal position relative to its current state.
* **Polar Sampling:** To ensure uniform coverage around the robot, goals are sampled using polar coordinates ($r, \theta$) and converted to Cartesian.
    * Radius $r \in [1.0, 5.0]$ meters.
    * The goal is always spawned at a fixed height relative to the floor ($z=0.5$).
* **Time Awareness:** The policy is explicitly conditioned on the **remaining time** in the episode, allowing it to learn "pacing" strategies (e.g., rushing if time is low, moving carefully if time is ample).

### 2. Custom Reward Structure
The reward system is designed to avoid over-constraining the motion (e.g., no strict velocity tracking penalty).

* **`task_reward` (Sparse-ish):** A dense signal $r = \frac{1}{1 + ||error||^2}$ that is **only activated** during the final seconds of the episode (e.g., last 1.0s). This forces the robot to be at the goal *at the end*, but allows exploration during the episode.
* **`explore` (Dense):** A cosine-similarity reward ($\frac{\mathbf{v} \cdot \mathbf{d}}{||\mathbf{v}|| ||\mathbf{d}||}$) that encourages moving in the general direction of the goal at all times.
* **`stalling` (Gated Penalty):** Penalizes the robot for standing still ($|v| < 0.1$) while far from the goal.
    * **Auto-Gating:** This penalty automatically deactivates once the agent learns the task (average task reward > 0.5), preventing it from interfering with fine-tuned terminal maneuvering.

### 3. Custom Curriculum
* **`terrain_levels_pos`:** Terrain difficulty increases based on **success** (distance to goal < 0.5m) rather than distance walked. This ensures the robot only faces harder terrain once it can reliably navigate to targets on easier terrain.

---

## 📂 Project Structure

The core logic is located in `source/IsaacLab_Terrains/IsaacLab_Terrains/tasks/manager_based/locomotion/position`:

```text
position/
├── config/
│   ├── anymal_c/          # Robot-specific configurations
│   │   ├── flat_env_cfg.py
│   │   └── rough_env_cfg.py
│   └── ...
├── mdp/                   # Markov Decision Process components
│   ├── commands.py        # UniformPose3dPolarCommand, TimeRemainingCommand
│   ├── observations.py    # Custom observation handlers
│   ├── rewards.py         # get_to_pos_in_time, exploration_incentive, stalling_penalty
│   ├── curriculums.py     # terrain_levels_pos (Goal-based curriculum)
│   └── ...
└── position_env_cfg.py    # Main Environment Configuration