import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import cv2
import numpy as np
import torch


def _get_bool_env(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_int_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float_env(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _setup_offscreen_camera(env):
    camera_props = gymapi.CameraProperties()
    camera_props.width = _get_int_env("LEGGED_GYM_VIDEO_WIDTH", 1280)
    camera_props.height = _get_int_env("LEGGED_GYM_VIDEO_HEIGHT", 720)
    camera_props.horizontal_fov = _get_float_env("LEGGED_GYM_VIDEO_FOV", 75.0)

    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    return camera_handle, camera_props.width, camera_props.height


def _get_camera_pose(env):
    robot_pos = env.base_pos[0].detach().cpu().numpy()
    if hasattr(env, "ball_pos"):
        target_pos = env.ball_pos[0].detach().cpu().numpy()
    else:
        target_pos = robot_pos + np.array([0.8, 0.0, 0.0], dtype=np.float32)

    lookat = 0.55 * robot_pos + 0.45 * target_pos
    lookat[2] = max(robot_pos[2], target_pos[2]) + 0.15

    camera_pos = lookat + np.array([-1.8, -1.6, 0.9], dtype=np.float32)
    return camera_pos, lookat


def _update_camera_pose(env, camera_handle):
    camera_pos, lookat = _get_camera_pose(env)
    env.gym.set_camera_location(
        camera_handle,
        env.envs[0],
        gymapi.Vec3(*camera_pos.tolist()),
        gymapi.Vec3(*lookat.tolist()),
    )


def _update_viewer_camera(env):
    camera_pos, lookat = _get_camera_pose(env)
    env.set_camera(camera_pos.tolist(), lookat.tolist())


def _get_camera_frame(env, camera_handle, width, height):
    if env.device != 'cpu':
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[0],
        camera_handle,
        gymapi.IMAGE_COLOR,
    )
    image = np.asarray(image, dtype=np.uint8)
    image = image.reshape(height, width, -1)
    return image[:, :, :3]


def _write_viewer_frame(env, frame_path):
    env.render(sync_frame_time=False)
    env.gym.write_viewer_image_to_file(env.viewer, str(frame_path))


def _build_video_output_paths(args):
    video_root = Path(args.video_dir or os.getenv("LEGGED_GYM_VIDEO_DIR", os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.experiment_name or args.task, "videos")))
    video_name = args.video_name or os.getenv("LEGGED_GYM_VIDEO_NAME", f"{args.task}_play")
    video_format = (args.video_format or os.getenv("LEGGED_GYM_VIDEO_FORMAT", "mp4")).lower().lstrip(".")
    if video_format != "mp4":
        print(f"Streaming recorder uses mp4 output; ignoring requested format: {video_format}")
        video_format = "mp4"
    temp_frame_path = video_root / f"{video_name}_stream_frame.png"
    video_path = video_root / f"{video_name}.{video_format}"
    return temp_frame_path, video_path


class StreamingVideoWriter:
    def __init__(self, video_path, fps):
        self.video_path = video_path
        self.fps = fps
        self.writer = None

    def append_data(self, frame, rgb=True):
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        if self.writer is None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.video_path}")
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _open_video_writer(video_path, fps):
    return StreamingVideoWriter(video_path, fps)


def _append_viewer_frame(env, writer, temp_frame_path):
    _update_viewer_camera(env)
    _write_viewer_frame(env, temp_frame_path)
    frame = cv2.imread(str(temp_frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read viewer frame: {temp_frame_path}")
    writer.append_data(frame, rgb=False)


def _load_policy(env, train_cfg, args):
    jit_policy_path = os.getenv("LEGGED_GYM_JIT_POLICY_PATH")
    if jit_policy_path:
        print('Loading scripted policy from:', jit_policy_path)
        policy = torch.jit.load(jit_policy_path, map_location=env.device)
        policy = policy.to(env.device)
        policy.eval()
        return policy, train_cfg

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    return policy, train_cfg


def play(args):
    record_video = args.record_video or _get_bool_env("LEGGED_GYM_RECORD_VIDEO", False)
    if record_video and args.headless:
        os.environ.setdefault("LEGGED_GYM_OFFSCREEN_RENDER", "1")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else (1 if record_video else 100)
    env_cfg.noise.add_noise = False # 禁用噪声
    env_cfg.domain_rand.randomize_friction = False # 摩擦系数随机化
    env_cfg.domain_rand.push_robots = False # 对机器人施加外部扰动

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations() # 获得观测的环境信息
    obs[:,6] = 1.0
    obs[:,7] = 0.0
    obs[:,8] = 0.0
        
    obs_size = obs.size()

    print("Observation tensor size:", obs_size)

    policy, train_cfg = _load_policy(env, train_cfg, args)

    camera_handle = None
    camera_width = None
    camera_height = None
    temp_frame_path = None
    video_path = None
    video_writer = None
    frame_idx = 0
    frame_stride = max(1, args.video_frame_stride or _get_int_env("LEGGED_GYM_VIDEO_FRAME_STRIDE", 2))
    max_steps = args.video_steps or _get_int_env("LEGGED_GYM_VIDEO_STEPS", 10 * int(env.max_episode_length))
    video_fps = max(1, args.video_fps or _get_int_env("LEGGED_GYM_VIDEO_FPS", 25))
    if record_video:
        temp_frame_path, video_path = _build_video_output_paths(args)
        if env.viewer is None:
            camera_handle, camera_width, camera_height = _setup_offscreen_camera(env)
        video_writer = _open_video_writer(video_path, video_fps)
        print('Streaming video to:', video_path)
        print('Temporary viewer frame path:', temp_frame_path)

    # action_file_path = "/home/hu/csq/unitree_rl_gym/deploy/actions_sim.log"
    # obs_file_path = "/home/hu/csq/unitree_rl_gym/deploy/obs_sim.log"

    # 将观测数据追加到文件中
   # with open(obs_file_path, "a") as obs_file:
    #    obs_file.write(",".join(map(str, obs.cpu().detach().numpy()[0])) + "\n")

    try:
        for i in range(max_steps):
            actions = policy(obs.detach()) # 将张量obs从计算图中分离出来，避免梯度传播

            # with open(action_file_path, "a") as action_file:
            #     action_file.write(",".join(map(str, actions.cpu().detach().numpy()[0])) + "\n")
            
            obs, _, rews, dones, infos = env.step(actions.detach()) # 获得新的观测
            obs[:,6] = 1.0
            obs[:,7] = 0.0
            obs[:,8] = 0.0

            if record_video and (i % frame_stride == 0):
                if env.viewer is not None:
                    _append_viewer_frame(env, video_writer, temp_frame_path)
                else:
                    _update_camera_pose(env, camera_handle)
                    frame = _get_camera_frame(env, camera_handle, camera_width, camera_height)
                    video_writer.append_data(frame)
                frame_idx += 1
            
            # 将观测数据追加到文件中
    #        with open(obs_file_path, "a") as obs_file:
       #         obs_file.write(",".join(map(str, obs.cpu().detach().numpy()[0])) + "\n")
    except KeyboardInterrupt:
        print('\nReceived KeyboardInterrupt. Finalizing video...')
    finally:
        if record_video and video_writer is not None:
            print('Recorded frame count:', frame_idx)
            video_writer.close()
            if temp_frame_path is not None and temp_frame_path.exists():
                temp_frame_path.unlink()
            print('Saved video to:', video_path)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    play(args)
