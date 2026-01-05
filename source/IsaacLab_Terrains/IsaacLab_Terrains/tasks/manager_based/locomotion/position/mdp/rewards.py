# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
import IsaacLab_Terrains.tasks.manager_based.locomotion.position.mdp as mdp


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def progress_toward_goal(env, command_name: str) -> torch.Tensor:
    """Reward for moving closer to the goal compared to the previous step."""

    cmd_term = env.command_manager.get_term(command_name)
    target_pos_b = cmd_term.command[:, :2] # (x, y) in base frame

    robot = env.scene["robot"]
    vel_b = robot.data.root_lin_vel_b[:, :2]

    dist = torch.norm(target_pos_b, dim=1, keepdim=True)
    dir_to_goal = target_pos_b / (dist + 1e-6)

    # how much of our velocity is towards the goal?
    progress = torch.sum(vel_b * dir_to_goal, dim=1)

    return progress


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""

    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w[:, :2]

    cmd_term = env.command_manager.get_term(command_name)

    target_pos_w = cmd_term.pose_command_w[:, :2]

    start_pos_w = env.scene.env_origins[:, :2]

    total_mission_dist = torch.norm(target_pos_w - start_pos_w, dim=1)
    dist_to_goal = torch.norm(target_pos_w - root_pos_w, dim=1)
    distance = dist_to_goal / total_mission_dist * 3.0
    # print(f"DISTANCE: {distance}")
    return 1 - torch.tanh(distance / std)


def get_to_pos_in_time(
    env: ManagerBasedRLEnv,
    reward_duration: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    robot_pos_w = robot.data.root_pos_w[:, :3]
    robot_start_pos_w = env.scene.env_origins[:, :3]
    goal_pos_b = cmd_term.pose_command_b[:, :3]

    robot_pos_b = robot_pos_w - robot_start_pos_w

    remaining_time = mdp.remaining_time_s(env)
    time_is_enough = torch.squeeze(remaining_time < reward_duration)

    error = torch.norm(robot_pos_b - goal_pos_b, dim=1)
    reward = 1.0 / ( 1.0 + error**2 ) / reward_duration

    return reward * time_is_enough


def exploration_incentive(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    robot_pos_w = robot.data.root_pos_w[:, :3]
    robot_start_pos_w = env.scene.env_origins[:, :3]
    robot_vel_w = robot.data.root_lin_vel_b[:, :3]
    goal_pos_b = cmd_term.pose_command_b[:, :3]

    robot_pos_b = robot_pos_w - robot_start_pos_w
    pos_error = goal_pos_b - robot_pos_b

    numerator = torch.sum(robot_vel_w * pos_error, dim=1)
    denominator = torch.mul(torch.linalg.vector_norm(robot_vel_w, dim=1), torch.linalg.vector_norm(pos_error, dim=1))
    reward = torch.div(numerator, denominator + 1e-6)
    return reward


def stalling_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    reward_duration: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    robot_pos_w = robot.data.root_pos_w[:, :3]
    robot_start_pos_w = env.scene.env_origins[:, :3]
    robot_vel_b = robot.data.root_lin_vel_b[:, :3]
    goal_pos_b = cmd_term.pose_command_b[:, :3]

    robot_pos_b = robot_pos_w - robot_start_pos_w
    # print(f"pos_b: {robot_pos_b[15]}")
    # print(f"goal_b: {goal_pos_b[15]}")
    # print("-"*20)

    task_val = get_to_pos_in_time(
        env,
        reward_duration=reward_duration,
        command_name=command_name,
        asset_cfg=asset_cfg
    )

    # if the task reward reaches 50% of its max, zero this reward
    if torch.mean(task_val) > 0.5:
        return torch.zeros(env.scene.num_envs, device="cuda")
    else:
        is_slow = torch.norm(robot_vel_b, dim=1) < 0.1
        is_far = torch.norm(robot_pos_b - goal_pos_b, dim=1) > 0.5
        reward = torch.ones(env.scene.num_envs, device="cuda")

        is_zero_reward = ~(is_slow & is_far)
        return reward * is_zero_reward