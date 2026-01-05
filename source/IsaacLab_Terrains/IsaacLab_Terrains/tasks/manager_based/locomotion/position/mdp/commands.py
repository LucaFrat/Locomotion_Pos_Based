from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import math
from collections.abc import Sequence



from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg, UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse
from isaaclab.managers import CommandTerm, CommandTermCfg

import IsaacLab_Terrains.tasks.manager_based.locomotion.position.mdp as mdp



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



class UniformPose2dPolarCommand(UniformPose2dCommand):

    cfg: UniformPose2dPolarCommandCfg

    def __init__(self, cfg: UniformPose2dPolarCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_cfg.name]

    def _resample_command(self, env_ids: torch.Tensor):
        robot_pos_w = self.robot.data.root_pos_w[env_ids, :2]

        r_min, r_max = self.cfg.radius_range
        r_squared = torch.rand(len(env_ids), device=self.device) * (r_max**2 - r_min**2) + r_min**2
        r = torch.sqrt(r_squared)

        theta_range = self.cfg.heading_range
        theta = torch.rand(len(env_ids), device=self.device) * (theta_range[1] - theta_range[0]) + theta_range[0]

        offset_x = r * torch.cos(theta)
        offset_y = r * torch.sin(theta)

        self.pos_command_w[env_ids, 0] = robot_pos_w[:, 0] + offset_x
        self.pos_command_w[env_ids, 1] = robot_pos_w[:, 1] + offset_y

        self.heading_command_w[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi) - torch.pi


@dataclass
class UniformPose2dPolarCommandCfg(UniformPose2dCommandCfg):
    class_type: type = UniformPose2dPolarCommand

    asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))

    radius_range: tuple[float, float] = (1.0, 5.0)
    heading_range: tuple[float, float] = (-3.14159, 3.14159)





class UniformPose3dPolarCommand(UniformPoseCommand):

    cfg: UniformPose3dPolarCommandCfg

    def __init__(self, cfg: UniformPose3dPolarCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_cfg.name]

    def _resample_command(self, env_ids: torch.Tensor):

        r_min, r_max = self.cfg.radius_range
        r_squared = torch.rand(len(env_ids), device=self.device) * (r_max**2 - r_min**2) + r_min**2
        r = torch.sqrt(r_squared)

        theta_range = self.cfg.heading_range
        theta = torch.rand(len(env_ids), device=self.device) * (theta_range[1] - theta_range[0]) + theta_range[0]

        offset_x = r * torch.cos(theta)
        offset_y = r * torch.sin(theta)

        self.pose_command_b[env_ids, 0] = offset_x
        self.pose_command_b[env_ids, 1] = offset_y
        self.pose_command_b[env_ids, 2] = 0.5

@dataclass
class UniformPose3dPolarCommandCfg(UniformPoseCommandCfg):
    class_type: type = UniformPose3dPolarCommand

    asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))

    radius_range: tuple[float, float] = (1.0, 5.0)
    heading_range: tuple[float, float] = (-3.14159, 3.14159)



class TimeRemainingCommand(CommandTerm):

    cfg: TimeRemainingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TimeRemainingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env

    """
    Properties
    """

    @property
    def command(self):
        return mdp.remaining_time_s(self.env)


    def _update_metrics(self):
        pass
    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass



@dataclass
class TimeRemainingCommandCfg(CommandTermCfg):
    class_type: type = TimeRemainingCommand