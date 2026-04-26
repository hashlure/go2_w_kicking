# go2w_rl_gym 三个环境说明

本文档总结 `/data1/yangqinze/U_G_W/go2w_rl_gym` 中当前注册的三个环境：

- `go2w`
- `go2w_walk_pretrain`
- `go2w_kick`

三个任务都基于 `legged_gym + rsl_rl + Isaac Gym`，入口注册在：

```text
go2w_rl_gym/legged_gym/envs/__init__.py
```

## 1. 环境总览

| 任务名 | 环境类 | 配置类 | 主要用途 |
| --- | --- | --- | --- |
| `go2w` | `Go2w` | `GO2WRoughCfg` | 基础 Go2W 轮足运动训练，主要在粗糙地形/台阶上学习前向运动 |
| `go2w_walk_pretrain` | `Go2wWalkPretrain` | `Go2wWalkPretrainCfg` | 平地行走预训练，观测维度与踢球任务对齐，方便后续迁移 |
| `go2w_kick` | `Go2wKick` | `Go2wKickCfg` | 踢球任务，机器人根据球的位置和速度学习接近、踢球、让球向前滚远 |

## 2. 共同控制逻辑

三个环境的动作维度都是 16，对应 Go2W 的 12 个腿部关节和 4 个轮足关节。

控制逻辑在：

```text
go2w_rl_gym/legged_gym/envs/go2w/go2w_robot.py
```

动作解释方式：

- 腿部关节：位置 PD 控制，目标位置近似为 `default_dof_pos + action * action_scale`
- 轮子关节：速度控制思想，轮子的 action 会通过 `vel_scale` 转成速度参考

关键参数：

```python
action_scale = 0.25
vel_scale = 10.0
decimation = 4
```

因此策略输出的 16 维动作中，腿部更像“目标角度增量”，轮子更像“目标转速输入”。

## 3. `go2w`

### 用途

`go2w` 是基础轮足 locomotion 环境，主要训练机器人在粗糙地形上运动。当前配置中，地形是 `trimesh`，且地形比例为 100% 上台阶：

```python
terrain_proportions = [0, 0, 1.0, 0, 0]
```

### 训练命令

```bash
cd /data1/yangqinze/U_G_W/go2w_rl_gym

python legged_gym/scripts/train.py \
  --task=go2w \
  --headless
```

### Play 命令

```bash
python legged_gym/scripts/play.py \
  --task=go2w \
  --num_envs=1
```

无头服务器保存视频：

```bash
python legged_gym/scripts/play.py \
  --task=go2w \
  --headless \
  --record_video \
  --num_envs=1 \
  --video_name=go2w_play \
  --video_steps=1000
```

### 命令输入

`go2w` 训练时使用随机速度命令：

```python
lin_vel_x = [0, 2]
lin_vel_y = [0, 0]
ang_vel_yaw = [0, 0]
```

也就是说，基础 `go2w` 当前主要学习 **前向速度跟踪**，不训练横向速度和 yaw 旋转。

### Actor 观测

`go2w` 的 actor 观测是 73 维：

| 维度范围 | 内容 | 维度 |
| --- | --- | --- |
| `0:3` | `base_ang_vel * 0.25`，机身角速度 | 3 |
| `3:6` | `projected_gravity`，机身坐标系下重力方向 | 3 |
| `6:9` | `commands[:3] * [2.0, 2.0, 0.25]` | 3 |
| `9:25` | `dof_err`，当前关节位置减默认关节位置 | 16 |
| `25:41` | `dof_vel * 0.05` | 16 |
| `41:57` | `dof_pos`，当前关节位置，轮子位置置零 | 16 |
| `57:73` | `actions`，上一帧/当前动作缓存 | 16 |

总计：

```text
3 + 3 + 3 + 16 + 16 + 16 + 16 = 73
```

### Critic privileged observation

`go2w` 的 privileged obs 会额外加入：

- `base_lin_vel * 2.0`
- 地形高度采样 `heights`

配置中 `num_privileged_obs = 263`。

## 4. `go2w_walk_pretrain`

### 用途

`go2w_walk_pretrain` 是平地行走预训练任务。它和 `go2w_kick` 保持同样的 actor 输入维度 79，但没有真实球，因此最后 6 维球信息填 0。

这样做的目的通常是：

- 先训练一个稳定行走/移动的基础策略
- 网络输入维度与踢球任务一致
- 后续可以更方便迁移到 `go2w_kick`

### 训练命令

```bash
cd /data1/yangqinze/U_G_W/go2w_rl_gym

python legged_gym/scripts/train.py \
  --task=go2w_walk_pretrain \
  --headless
```

指定迭代数：

```bash
python legged_gym/scripts/train.py \
  --task=go2w_walk_pretrain \
  --headless \
  --max_iterations=3000
```

### Play 命令

使用最新 checkpoint：

```bash
python legged_gym/scripts/play.py \
  --task=go2w_walk_pretrain \
  --num_envs=1
```

使用指定权重：

```bash
python legged_gym/scripts/play.py \
  --task=go2w_walk_pretrain \
  --headless \
  --record_video \
  --num_envs=1 \
  --checkpoint=/data1/yangqinze/U_G_W/go2w_rl_gym/logs/go2w_walk_pretrain/Apr26_08-54-54_/model_3000.pt \
  --video_name=go2w_walk_model_3000 \
  --video_steps=1000
```

默认视频输出：

```text
go2w_rl_gym/logs/go2w_walk_pretrain/videos/go2w_walk_model_3000.mp4
```

### 命令输入

`go2w_walk_pretrain` 训练时使用随机速度命令：

```python
lin_vel_x = [-0.5, 1.5]
lin_vel_y = [-0.5, 0.5]
ang_vel_yaw = [-1.0, 1.0]
```

命令每 10 秒重新采样一次：

```python
resampling_time = 10.0
```

命令输入到 actor 前会缩放：

```python
[lin_vel_x * 2.0, lin_vel_y * 2.0, ang_vel_yaw * 0.25]
```

### Actor 观测

`go2w_walk_pretrain` 的 actor 观测是 79 维：

| 维度范围 | 内容 | 维度 |
| --- | --- | --- |
| `0:3` | `base_ang_vel * 0.25` | 3 |
| `3:6` | `projected_gravity` | 3 |
| `6:9` | `commands[:3] * [2.0, 2.0, 0.25]` | 3 |
| `9:25` | `dof_err`，轮子位置误差置零 | 16 |
| `25:41` | `dof_vel * 0.05` | 16 |
| `41:57` | `dof_pos_obs`，轮子位置置零 | 16 |
| `57:73` | `actions` | 16 |
| `73:76` | `zero_ball_pos_body`，全 0 | 3 |
| `76:79` | `zero_ball_vel_body`，全 0 | 3 |

总计：

```text
3 + 3 + 3 + 16 + 16 + 16 + 16 + 3 + 3 = 79
```

### 和 `go2w` 的区别

相比 `go2w`，`go2w_walk_pretrain` 主要变化：

- 地形从台阶/粗糙地形改为平地 `plane`
- 随机命令更丰富，包含前后、左右、yaw 旋转
- actor 观测从 73 维增加到 79 维
- 增加最后 6 维全 0 的球信息，用来和踢球任务对齐
- `num_privileged_obs` 从 `263` 改为 `82`

## 5. `go2w_kick`

### 用途

`go2w_kick` 是踢球任务。机器人不再学习随机速度跟踪，而是根据球相对自己的位置和速度，学习：

1. 靠近球
2. 让脚靠近球
3. 踢中球
4. 让球沿 +x 方向滚远
5. 踢完后和球分离

### 训练命令

```bash
cd /data1/yangqinze/U_G_W/go2w_rl_gym

python legged_gym/scripts/train.py \
  --task=go2w_kick \
  --headless
```

指定迭代数：

```bash
python legged_gym/scripts/train.py \
  --task=go2w_kick \
  --headless \
  --max_iterations=10000
```

### Play 命令

```bash
python legged_gym/scripts/play.py \
  --task=go2w_kick \
  --headless \
  --record_video \
  --num_envs=1 \
  --video_name=go2w_kick_play \
  --video_steps=1000
```

### 球初始化

球参数：

```python
radius = 0.11
mass = 0.43
friction = 0.8
restitution = 0.35
```

每次 reset 时，球放在机器人前方：

```python
init_x_range = [0.7, 1.0]
init_y_range = [-0.25, 0.25]
```

### 命令输入

`go2w_kick` 中速度命令全部为 0：

```python
lin_vel_x = [0.0, 0.0]
lin_vel_y = [0.0, 0.0]
ang_vel_yaw = [0.0, 0.0]
```

所以训练时 actor 观测里的 command 维度基本恒为 0。踢球策略主要依赖球的相对位置和速度，而不是外部速度指令。

### Actor 观测

`go2w_kick` 的 actor 观测同样是 79 维：

| 维度范围 | 内容 | 维度 |
| --- | --- | --- |
| `0:3` | `base_ang_vel * 0.25` | 3 |
| `3:6` | `projected_gravity` | 3 |
| `6:9` | `commands[:3] * [2.0, 2.0, 0.25]`，基本为 0 | 3 |
| `9:25` | `dof_err`，轮子位置误差置零 | 16 |
| `25:41` | `dof_vel * 0.05` | 16 |
| `41:57` | `dof_pos_obs`，轮子位置置零 | 16 |
| `57:73` | `actions` | 16 |
| `73:76` | `ball_pos_body`，球在机器人坐标系下的相对位置 | 3 |
| `76:79` | `ball_vel_body`，球在机器人坐标系下的速度 | 3 |

与 `go2w_walk_pretrain` 的区别是：

- `go2w_walk_pretrain` 最后 6 维是全 0
- `go2w_kick` 最后 6 维是真实球信息

### Critic privileged observation

critic 观测是 82 维：

```text
actor_obs 79维 + base_lin_vel * 2.0 3维 = 82维
```

也就是说 actor 看不到真实 base 线速度，critic 可以看到。

### 踢球状态判断

每一步会计算脚和球的距离：

```python
feet_ball_dist = min(distance(feet, ball))
```

接触球：

```python
contact_ball = feet_ball_dist < ball.radius + contact_margin
```

有效踢球：

```python
valid_kick = contact_ball and ball_forward_vel > 0.6
```

第一次有效踢球：

```python
new_kick = valid_kick and not has_kicked
```

### 奖励函数

踢球相关正奖励：

| 奖励项 | 权重 | 含义 |
| --- | ---: | --- |
| `approach_ball` | `0.3` | 踢球前鼓励机器人靠近球 |
| `feet_to_ball` | `0.4` | 踢球前鼓励脚靠近球 |
| `kick_once` | `8.0` | 第一次有效踢球给大奖励 |
| `ball_forward_vel` | `3.0` | 踢中后鼓励球的 +x 方向速度 |
| `ball_forward_dist` | `2.0` | 踢中后鼓励球向前滚远 |
| `separate_after_kick` | `2.0` | 踢完后鼓励机器人和球分离 |
| `success` | `10.0` | 成功完成踢球任务 |

踢球相关负奖励：

| 奖励项 | 权重 | 含义 |
| --- | ---: | --- |
| `stay_close_after_kick` | `-2.0` | 踢完后仍贴近球会扣分 |
| `long_contact_ball` | `-0.5` | 长时间接触球会扣分，避免推球/夹球 |
| `rear_leg_ground_contact` | `-0.8` | 初期后腿触地扣分，抑制不自然姿态 |

稳定性和能耗相关奖励/惩罚：

| 奖励项 | 权重 |
| --- | ---: |
| `tracking_lin_vel` | `0.2` |
| `tracking_ang_vel` | `0.1` |
| `termination` | `-0.8` |
| `lin_vel_z` | `-0.1` |
| `ang_vel_xy` | `-0.1` |
| `orientation` | `-2.0` |
| `torques` | `-0.0002` |
| `dof_vel` | `-1e-7` |
| `dof_acc` | `-1e-7` |
| `base_height` | `-0.8` |
| `collision` | `-0.2` |
| `feet_stumble` | `-0.1` |
| `action_rate` | `-0.001` |
| `dof_pos_limits` | `-0.8` |
| `hip_action_l2` | `-0.2` |

注意：这些 scale 在 legged_gym 中通常会乘以 `dt` 后用于每步奖励，所以表里的数值主要表示相对权重。

### 成功条件

成功条件：

```python
forward_dist > 1.0
has_kicked == True
robot_ball_dist > 0.40
```

含义：

- 球向 +x 方向滚出超过 1.0 米
- 发生过有效踢球
- 机器人和球距离大于 0.40 米，说明踢完后已经分离

### Reset 条件

环境会在以下情况 reset：

- episode 超时
- base 太低，疑似倒地/接触地面
- 成功完成踢球
- 球离环境原点超过 3.0 米
- 球飞起高度超过 0.6 米

### 训练逻辑总结

`go2w_kick` 的整体训练逻辑可以理解为：

```text
reset:
  球随机放在机器人前方 0.7m 到 1.0m

每一步:
  actor 看到自身状态 + 球相对位置/速度
  actor 输出 16 维动作
  环境执行轮足控制
  更新球和机器人状态
  判断是否接触球、是否有效踢球、是否第一次踢球
  计算奖励
  判断是否成功或 reset

奖励引导:
  踢前靠近球
  脚靠近球
  第一次踢中奖励很大
  踢中后奖励球向前速度和前进距离
  踢完后奖励机器人离开球
```

## 6. 常用日志和模型路径

训练日志默认保存到：

```text
go2w_rl_gym/logs/<experiment_name>/<date_time>_<run_name>/
```

例如：

```text
go2w_rl_gym/logs/go2w_walk_pretrain/Apr26_08-54-54_/model_3000.pt
```

Play 时导出的 JIT policy 默认保存到：

```text
go2w_rl_gym/logs/<experiment_name>/exported/policy_1.pt
```

## 7. 常用命令速查

### 训练

```bash
python legged_gym/scripts/train.py --task=go2w --headless
python legged_gym/scripts/train.py --task=go2w_walk_pretrain --headless
python legged_gym/scripts/train.py --task=go2w_kick --headless
```

### 回放

```bash
python legged_gym/scripts/play.py --task=go2w --num_envs=1
python legged_gym/scripts/play.py --task=go2w_walk_pretrain --num_envs=1
python legged_gym/scripts/play.py --task=go2w_kick --num_envs=1
```

### 无头服务器录视频

```bash
python legged_gym/scripts/play.py \
  --task=go2w_walk_pretrain \
  --headless \
  --record_video \
  --num_envs=1 \
  --checkpoint=/data1/yangqinze/U_G_W/go2w_rl_gym/logs/go2w_walk_pretrain/Apr26_08-54-54_/model_3000.pt \
  --video_name=go2w_walk_model_3000 \
  --video_steps=1000
```

默认输出：

```text
go2w_rl_gym/logs/<experiment_name>/videos/<video_name>.mp4
```

## 8. 三个环境的核心区别

| 项目 | `go2w` | `go2w_walk_pretrain` | `go2w_kick` |
| --- | --- | --- | --- |
| 地形 | 台阶/粗糙地形 | 平地 | 平地 |
| actor obs | 73 | 79 | 79 |
| privileged obs | 263 | 82 | 82 |
| 是否有球 | 否 | 否，球信息填 0 | 是 |
| command | 前向速度随机 | 前后/左右/yaw 随机 | 全 0 |
| 主要目标 | 前向运动/地形通过 | 平地速度跟踪预训练 | 踢球成功 |
| 训练时长配置 | 30000 iter | 3000 iter | 10000 iter |

