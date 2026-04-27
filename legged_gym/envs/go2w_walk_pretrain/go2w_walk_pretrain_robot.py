import torch

from legged_gym.envs.go2w.go2w_robot import Go2w

from .go2w_walk_pretrain_config import Go2wWalkPretrainCfg


class Go2wWalkPretrain(Go2w):
    cfg: Go2wWalkPretrainCfg

    def compute_observations(self):
        self.base_height_command = torch.tensor(
            self.cfg.rewards.base_height_target,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:, self.wheel_indices] = 0.0

        dof_pos_obs = self.dof_pos.clone()
        dof_pos_obs[:, self.wheel_indices] = 0.0

        zero_ball_pos_body = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_err * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                dof_pos_obs,
                self.actions,
                zero_ball_pos_body,
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
