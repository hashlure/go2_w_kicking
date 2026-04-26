from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class Go2wWalkPretrainCfg(GO2WRoughCfg):
    class env(GO2WRoughCfg.env):
        num_observations = 79
        num_privileged_obs = 82
        num_actions = 16
        episode_length_s = 10.0

    class terrain(GO2WRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(GO2WRoughCfg.commands):
        curriculum = True
        heading_command = False
        resampling_time = 10.0

        class ranges(GO2WRoughCfg.commands.ranges):
            lin_vel_x = [-0.5, 1.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]

    class rewards(GO2WRoughCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.4
        base_height_target = 0.34

        class scales(GO2WRoughCfg.rewards.scales):
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            orientation = -2.0
            lin_vel_z = -0.1
            ang_vel_xy = -0.05
            base_height = -0.5
            collision = -0.1
            torques = -0.0001
            dof_vel = -1e-7
            dof_acc = -1e-7
            action_rate = -0.0002
            dof_pos_limits = -0.9
            hip_action_l2 = -0.1
            stand_still = -0.01
            feet_air_time = 0.0
            feet_stumble = -0.1
            termination = -0.8
            approach_ball = 0.0
            feet_to_ball = 0.0
            kick_once = 0.0
            ball_forward_vel = 0.0
            ball_forward_dist = 0.0
            separate_after_kick = 0.0
            success = 0.0
            stay_close_after_kick = 0.0
            long_contact_ball = 0.0
            rear_leg_ground_contact = 0.0


class Go2wWalkPretrainCfgPPO(GO2WRoughCfgPPO):
    class runner(GO2WRoughCfgPPO.runner):
        experiment_name = "go2w_walk_pretrain"
        run_name = ""
        max_iterations = 3000