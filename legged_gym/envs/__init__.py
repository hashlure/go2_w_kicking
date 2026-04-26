from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO
from legged_gym.envs.go2w_kick.go2w_kick_config import Go2wKickCfg, Go2wKickCfgPPO
from legged_gym.envs.go2w_walk_pretrain.go2w_walk_pretrain_config import Go2wWalkPretrainCfg, Go2wWalkPretrainCfgPPO
from .base.legged_robot import LeggedRobot
from .go2w.go2w_robot import Go2w
from .go2w_kick.go2w_kick_robot import Go2wKick
from .go2w_walk_pretrain.go2w_walk_pretrain_robot import Go2wWalkPretrain

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2w", Go2w, GO2WRoughCfg(), GO2WRoughCfgPPO())
task_registry.register( "go2w_kick", Go2wKick, Go2wKickCfg(), Go2wKickCfgPPO())
task_registry.register( "go2w_walk_pretrain", Go2wWalkPretrain, Go2wWalkPretrainCfg(), Go2wWalkPretrainCfgPPO())
