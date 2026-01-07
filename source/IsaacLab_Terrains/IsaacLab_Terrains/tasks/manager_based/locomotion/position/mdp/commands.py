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
import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import CUBOID_MARKER_CFG

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

        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]

        self.pos_command_w[env_ids, 0] += offset_x
        self.pos_command_w[env_ids, 1] += offset_y
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        self.heading_command_w[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi) - torch.pi

@dataclass
class UniformPose2dPolarCommandCfg(UniformPose2dCommandCfg):
    class_type: type = UniformPose2dPolarCommand

    asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))

    radius_range: tuple[float, float] = (1.0, 5.0)
    heading_range: tuple[float, float] = (-3.14159, 3.14159)




class UniformPose3dPolarCommand(UniformPose2dCommand):
    """
    Generates commands in polar coordinates and samples the Z-height
    directly from the physics terrain mesh.
    """
    cfg: UniformPose3dPolarCommandCfg

    def __init__(self, cfg: UniformPose3dPolarCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_cfg.name]

        # Cache the terrain generator if available
        self.terrain_generator = None
        if hasattr(env.scene.terrain, "terrain_generator"):
            self.terrain_generator = env.scene.terrain.terrain_generator

    def _resample_command(self, env_ids: torch.Tensor):
        # 1. Sample Radius and Angle
        # We sample on the same device as the simulation
        r_min, r_max = self.cfg.radius_range
        r_squared = torch.rand(len(env_ids), device=self.device) * (r_max**2 - r_min**2) + r_min**2
        r = torch.sqrt(r_squared)

        theta_range = self.cfg.heading_range
        theta = torch.rand(len(env_ids), device=self.device) * (theta_range[1] - theta_range[0]) + theta_range[0]

        offset_x = r * torch.cos(theta)
        offset_y = r * torch.sin(theta)

        # 2. Determine World X, Y
        # Start at environment origin (center of the terrain tile)
        # Note: We use env_origins for X,Y, but we will calculate Z manually
        env_origins = self._env.scene.env_origins[env_ids]

        target_x = env_origins[:, 0] + offset_x
        target_y = env_origins[:, 1] + offset_y

        # Update internal command buffer (X, Y)
        self.pos_command_w[env_ids, 0] = target_x
        self.pos_command_w[env_ids, 1] = target_y

        # 3. Determine World Z (Ray Cast)
        # If we have a terrain generator with a mesh, we raycast to find the floor.
        if self.terrain_generator is not None and hasattr(self.terrain_generator, "terrain_mesh"):
            # Prepare Ray Origins: [x, y, high_z]
            # We start the ray high up (e.g., +100m) and cast down
            ray_origins = torch.zeros((len(env_ids), 3), device=self.device)
            ray_origins[:, 0] = target_x
            ray_origins[:, 1] = target_y
            ray_origins[:, 2] = env_origins[:, 2] + 5.0 # Start 5m above the env origin to be safe

            # Prepare Ray Directions: [0, 0, -1]
            ray_dirs = torch.zeros((len(env_ids), 3), device=self.device)
            ray_dirs[:, 2] = -1.0

            # Move to CPU for Trimesh (Trimesh operations are CPU based)
            # Note: For ~1-2k envs this is fast enough (ms range).
            # If extremely slow, consider Warp or PhysX query.
            origins_np = ray_origins.cpu().numpy()
            dirs_np = ray_dirs.cpu().numpy()

            # Perform Ray Cast
            # returns: locations, index_ray, index_tri
            # intersects_location allows multiple hits, we take the first hit per ray usually
            # But simpler: use ray.intersects_first if available, or process results
            locations, index_ray, _ = self.terrain_generator.terrain_mesh.ray.intersects_location(
                ray_origins=origins_np,
                ray_directions=dirs_np,
                multiple_hits=False
            )

            # Create a default Z buffer (default to env_origin Z if ray misses)
            target_z = env_origins[:, 2].clone()

            # Update the Z values for rays that hit the mesh
            if len(index_ray) > 0:
                # locations is (N_hits, 3). index_ray is (N_hits,) mapping back to our input array
                hit_z = torch.from_numpy(locations[:, 2]).to(self.device, dtype=torch.float)
                indices = torch.from_numpy(index_ray).to(self.device, dtype=torch.long)

                # Assign hit height
                target_z[indices] = hit_z

            # Set command Z = Floor Height + 0.5m
            self.pos_command_w[env_ids, 2] = target_z + 0.5

        else:
            # Fallback for flat terrain (no mesh generator)
            self.pos_command_w[env_ids, 2] = env_origins[:, 2] + 0.5

        # 4. Heading
        self.heading_command_w[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi) - torch.pi

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first time
    #     if debug_vis:
    #         if not hasattr(self, "goal_pose_visualizer"):
    #             self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
    #         # set their visibility to true
    #         self.goal_pose_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_pose_visualizer"):
    #             self.goal_pose_visualizer.set_visibility(False)


    # def _debug_vis_callback(self, event):
    #     # update the box marker
    #     self.goal_pose_visualizer.visualize(
    #         translations=self.pos_command_w,
    #         orientations=quat_from_euler_xyz(
    #             torch.zeros_like(self.heading_command_w),
    #             torch.zeros_like(self.heading_command_w),
    #             self.heading_command_w,
    #         ),
    #     )


@dataclass
class UniformPose3dPolarCommandCfg(UniformPose2dCommandCfg):
    class_type: type = UniformPose3dPolarCommand

    asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))

    radius_range: tuple[float, float] = (1.0, 5.0)
    heading_range: tuple[float, float] = (-3.14, 3.14)

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = field(default_factory=lambda: CUBOID_MARKER_CFG)


class TimeRemainingCommand(CommandTerm):

    cfg: TimeRemainingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TimeRemainingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.env = env

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