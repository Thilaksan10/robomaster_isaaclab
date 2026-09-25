import gymnasium as gym

from .robomaster import RobomasterEnv, RobomasterEnvCfg
from .robomaster_racetrack import RacetrackEnv, RacetrackEnvCfg
from .robomaster_edgecases import EdgeCasesEnv, EdgeCasesEnvCfg
import cfg_train

gym.register(
    id="Robomaster",
    entry_point=RobomasterEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RobomasterEnvCfg,
        "rl_games_cfg_entry_point": f"{cfg_train.__name__}:rl_games_ppo_cfg.yaml",

    }
)

gym.register(
    id="Racetrack",
    entry_point=RacetrackEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RacetrackEnvCfg,
        "rl_games_cfg_entry_point": f"{cfg_train.__name__}:rl_games_ppo_cfg_racetrack.yaml",

    }
)

gym.register(
    id="EdgeCases",
    entry_point=EdgeCasesEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EdgeCasesEnvCfg,
        "rl_games_cfg_entry_point": f"{cfg_train.__name__}:rl_games_ppo_cfg_edgecases.yaml",

    }
)