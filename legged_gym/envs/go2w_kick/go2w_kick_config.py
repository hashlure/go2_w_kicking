from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO


class Go2wKickCfg(GO2WRoughCfg):
    class env(GO2WRoughCfg.env):
        num_observations = 79
        num_privileged_obs = 82
        num_actions = 16
        episode_length_s = 6.0

    class terrain(GO2WRoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class commands(GO2WRoughCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 10.0

        class ranges(GO2WRoughCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class ball:
        radius = 0.11
        mass = 0.43
        friction = 0.8
        restitution = 0.35
        init_x_range = [0.7, 1.0]
        init_y_range = [-0.25, 0.25]
        max_forward_dist = 1.0
        max_distance = 3.0
        kick_vel_threshold = 0.6
        detach_dist = 0.40
        contact_margin = 0.06
        max_contact_steps = 5
        max_contact_steps_hard = 15

    class rewards(GO2WRoughCfg.rewards):
        only_positive_rewards = True

        class scales(GO2WRoughCfg.rewards.scales):
            tracking_lin_vel = 0.2
            tracking_ang_vel = 0.1
            feet_air_time = 0.0
            approach_ball = 0.3
            feet_to_ball = 0.4
            kick_once = 8.0
            ball_forward_vel = 3.0
            ball_forward_dist = 2.0
            separate_after_kick = 2.0
            success = 10.0
            stay_close_after_kick = -2.0
            long_contact_ball = -0.5
            rear_leg_ground_contact = -0.8
            termination = -0.8
            lin_vel_z = -0.1
            ang_vel_xy = -0.1
            orientation = -2.0
            torques = -0.0002
            dof_vel = -1e-7
            dof_acc = -1e-7
            base_height = -0.8
            collision = -0.2
            feet_stumble = -0.1
            action_rate = -0.001
            stand_still = 0.0
            dof_pos_limits = -0.8
            arm_pos = 0.0
            hip_action_l2 = -0.2


class Go2wKickCfgPPO(GO2WRoughCfgPPO):
    class runner(GO2WRoughCfgPPO.runner):
        experiment_name = "go2w_kick"
        run_name = ""
        max_iterations = 10000
        save_interval = 1000