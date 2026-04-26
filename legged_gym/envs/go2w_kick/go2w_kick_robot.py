import os
import numpy as np

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2w.go2w_robot import Go2w
from .go2w_kick_config import Go2wKickCfg


class Go2wKick(Go2w):
    cfg: Go2wKickCfg

    def _refresh_actor_root_views(self):
        self.root_states = self.root_states_all[self.robot_actor_indices]
        self.ball_root_states = self.root_states_all[self.ball_actor_indices]
        self.base_pos = self.root_states[:, 0:3]
        self.ball_pos = self.ball_root_states[:, 0:3]
        self.ball_quat = self.ball_root_states[:, 3:7]
        self.ball_lin_vel = self.ball_root_states[:, 7:10]
        self.ball_ang_vel = self.ball_root_states[:, 10:13]

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [name for name in body_names if self.cfg.asset.foot_name in name]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([body_name for body_name in body_names if name in body_name])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([body_name for body_name in body_names if name in body_name])
        wheel_names = []
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([dof_name for dof_name in self.dof_names if name in dof_name])

        ball_asset_options = gymapi.AssetOptions()
        ball_volume = (4.0 / 3.0) * np.pi * self.cfg.ball.radius ** 3
        ball_asset_options.density = self.cfg.ball.mass / ball_volume
        ball_asset_options.angular_damping = 0.01
        ball_asset_options.linear_damping = 0.01
        ball_asset_options.max_angular_velocity = 100.0
        ball_asset_options.max_linear_velocity = 100.0
        ball_asset_options.disable_gravity = False
        ball_asset = self.gym.create_sphere(self.sim, self.cfg.ball.radius, ball_asset_options)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.robot_actor_handles = []
        self.ball_actor_handles = []
        robot_actor_indices = []
        ball_actor_indices = []

        for env_id in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[env_id].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            robot_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                env_id,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, env_id)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props = self._process_rigid_body_props(body_props, env_id)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)

            ball_pose = gymapi.Transform()
            ball_pose.p = gymapi.Vec3(
                self.env_origins[env_id, 0].item() + 0.8,
                self.env_origins[env_id, 1].item(),
                self.cfg.ball.radius,
            )
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", env_id, 0, 0)
            ball_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
            for prop in ball_shape_props:
                prop.friction = self.cfg.ball.friction
                prop.restitution = self.cfg.ball.restitution
            self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)
            ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            for prop in ball_body_props:
                prop.mass = self.cfg.ball.mass
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
            self.robot_actor_handles.append(robot_handle)
            self.ball_actor_handles.append(ball_handle)
            robot_actor_indices.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))
            ball_actor_indices.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))

        self.robot_actor_indices = torch.tensor(robot_actor_indices, dtype=torch.long, device=self.device)
        self.ball_actor_indices = torch.tensor(ball_actor_indices, dtype=torch.long, device=self.device)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for index, name in enumerate(feet_names):
            self.feet_indices[index] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for index, name in enumerate(penalized_contact_names):
            self.penalised_contact_indices[index] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for index, name in enumerate(termination_contact_names):
            self.termination_contact_indices[index] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for index, name in enumerate(wheel_names):
            self.wheel_indices[index] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], name)

        rear_leg_contact_names = [
            name for name in body_names
            if name in ("RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf")
        ]
        self.rear_leg_contact_indices = torch.zeros(
            len(rear_leg_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for index, name in enumerate(rear_leg_contact_names):
            self.rear_leg_contact_indices[index] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

    def _init_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states_all = gymtorch.wrap_tensor(actor_root_state).view(-1, 13)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self._refresh_actor_root_views()

        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.measured_heights = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.init_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for index in range(self.num_dofs):
            name = self.dof_names[index]
            self.default_dof_pos[index] = self.cfg.init_state.default_joint_angles[name]
            self.init_dof_pos[index] = self.cfg.init_state.init_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[index] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[index] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[index] = 0.0
                self.d_gains[index] = 0.0
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.init_dof_pos = self.init_dof_pos.unsqueeze(0)

        self.ball_init_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.has_kicked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.kick_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ball_contact_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ball_max_forward_vel = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_ball = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.valid_kick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.new_kick = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def post_physics_step(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self._refresh_actor_root_views()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()
        self.contact_ball, self.valid_kick, self.new_kick = self._update_kick_state()
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        new_states = self.base_init_state.unsqueeze(0).repeat(len(env_ids), 1)
        new_states[:, :3] += self.env_origins[env_ids]
        if self.custom_origins:
            new_states[:, :2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_states[:, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        actor_ids = self.robot_actor_indices[env_ids]
        self.root_states_all[actor_ids] = new_states

        actor_ids_int32 = actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states_all),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )
        self._refresh_actor_root_views()

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        self.dof_pos[env_ids] = self.init_dof_pos
        self.dof_vel[env_ids] = 0.0

        actor_ids_int32 = self.robot_actor_indices[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )

    def _reset_ball_states(self, env_ids):
        if len(env_ids) == 0:
            return

        new_ball_states = torch.zeros(len(env_ids), 13, dtype=torch.float, device=self.device)
        ball_x = torch_rand_float(
            self.cfg.ball.init_x_range[0],
            self.cfg.ball.init_x_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        ball_y = torch_rand_float(
            self.cfg.ball.init_y_range[0],
            self.cfg.ball.init_y_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        new_ball_states[:, 0] = self.env_origins[env_ids, 0] + ball_x
        new_ball_states[:, 1] = self.env_origins[env_ids, 1] + ball_y
        new_ball_states[:, 2] = self.cfg.ball.radius
        new_ball_states[:, 6] = 1.0

        actor_ids = self.ball_actor_indices[env_ids]
        self.root_states_all[actor_ids] = new_ball_states
        self.ball_init_pos[env_ids] = new_ball_states[:, 0:3]
        self.has_kicked[env_ids] = False
        self.kick_step_buf[env_ids] = 0
        self.ball_contact_steps[env_ids] = 0.0
        self.ball_max_forward_vel[env_ids] = 0.0
        self.success_buf[env_ids] = False
        self.contact_ball[env_ids] = False
        self.valid_kick[env_ids] = False
        self.new_kick[env_ids] = False

        actor_ids_int32 = actor_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states_all),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )
        self._refresh_actor_root_views()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._reset_ball_states(env_ids)
        self._resample_commands(env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states_all[self.robot_actor_indices, 7:9] = torch_rand_float(
            -max_vel,
            max_vel,
            (self.num_envs, 2),
            device=self.device,
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_all))
        self._refresh_actor_root_views()

    def compute_observations(self):
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:, self.wheel_indices] = 0.0
        dof_pos_obs = self.dof_pos.clone()
        dof_pos_obs[:, self.wheel_indices] = 0.0

        ball_pos_body = quat_rotate_inverse(self.base_quat, self.ball_pos - self.base_pos)
        ball_vel_body = quat_rotate_inverse(self.base_quat, self.ball_lin_vel)

        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_err * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                dof_pos_obs,
                self.actions,
                ball_pos_body,
                ball_vel_body,
            ),
            dim=-1,
        )

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.privileged_obs_buf = torch.cat(
            (
                self.obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
            ),
            dim=-1,
        )

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = self.cfg.noise.add_noise
        return torch.zeros(self.cfg.env.num_observations, dtype=torch.float, device=self.device)

    def _get_feet_ball_dist(self):
        feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        dist = torch.norm(feet_pos[:, :, :2] - self.ball_pos[:, None, :2], dim=-1)
        min_dist, _ = torch.min(dist, dim=1)
        return min_dist

    def _get_robot_ball_dist(self):
        return torch.norm(self.base_pos[:, :2] - self.ball_pos[:, :2], dim=1)

    def _get_ball_forward_dist(self):
        return self.ball_pos[:, 0] - self.ball_init_pos[:, 0]

    def _update_kick_state(self):
        feet_ball_dist = self._get_feet_ball_dist()
        contact_ball = feet_ball_dist < (self.cfg.ball.radius + self.cfg.ball.contact_margin)
        ball_forward_vel = self.ball_lin_vel[:, 0]
        valid_kick = contact_ball & (ball_forward_vel > self.cfg.ball.kick_vel_threshold)
        new_kick = valid_kick & (~self.has_kicked)

        self.has_kicked |= new_kick
        self.kick_step_buf[new_kick] = self.episode_length_buf[new_kick]
        self.ball_max_forward_vel = torch.maximum(self.ball_max_forward_vel, ball_forward_vel)
        self.ball_contact_steps += contact_ball.float()
        return contact_ball, valid_kick, new_kick

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for index in range(len(self.reward_functions)):
            name = self.reward_names[index]
            rew = self.reward_functions[index]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        contact_flag = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.base_contact_buf = torch.any(contact_flag.unsqueeze(1) < 0.20, dim=1)
        self.reset_buf |= self.base_contact_buf

        forward_dist = self._get_ball_forward_dist()
        robot_ball_dist = self._get_robot_ball_dist()
        success = (
            (forward_dist > self.cfg.ball.max_forward_dist)
            & self.has_kicked
            & (robot_ball_dist > self.cfg.ball.detach_dist)
        )
        ball_dist_from_origin = torch.norm(self.ball_pos[:, :2] - self.env_origins[:, :2], dim=1)
        ball_out = ball_dist_from_origin > self.cfg.ball.max_distance
        ball_fly = self.ball_pos[:, 2] > 0.6

        self.success_buf = success
        self.reset_buf |= success
        self.reset_buf |= ball_out
        self.reset_buf |= ball_fly

    def _reward_approach_ball(self):
        dist = self._get_robot_ball_dist()
        return torch.exp(-2.0 * dist) * (~self.has_kicked).float()

    def _reward_feet_to_ball(self):
        feet_ball_dist = self._get_feet_ball_dist()
        return torch.exp(-10.0 * feet_ball_dist) * (~self.has_kicked).float()

    def _reward_kick_once(self):
        return self.new_kick.float()

    def _reward_ball_forward_vel(self):
        return torch.clamp(self.ball_lin_vel[:, 0], min=0.0, max=3.0) * self.has_kicked.float()

    def _reward_ball_forward_dist(self):
        forward_dist = self._get_ball_forward_dist()
        return torch.clamp(forward_dist, min=0.0, max=2.0) * self.has_kicked.float()

    def _reward_separate_after_kick(self):
        robot_ball_dist = self._get_robot_ball_dist()
        return self.has_kicked.float() * torch.clamp(robot_ball_dist - self.cfg.ball.detach_dist, min=0.0, max=1.0)

    def _reward_stay_close_after_kick(self):
        robot_ball_dist = self._get_robot_ball_dist()
        too_close = robot_ball_dist < self.cfg.ball.detach_dist
        return self.has_kicked.float() * too_close.float()

    def _reward_long_contact_ball(self):
        return torch.clamp(self.ball_contact_steps - self.cfg.ball.max_contact_steps, min=0.0)

    def _reward_rear_leg_ground_contact(self):
        rear_leg_contact = torch.norm(self.contact_forces[:, self.rear_leg_contact_indices, :], dim=-1) > 1.0
        early_phase = self.episode_length_buf < int(1.0 / self.dt)
        return torch.sum(rear_leg_contact.float(), dim=1) * early_phase.float()

    def _reward_success(self):
        return self.success_buf.float()