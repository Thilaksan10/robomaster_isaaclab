# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--optimized",
    action="store_true",
    default=False,
    help="Use the optimized simulation model.",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Explicit RL-Games experiment/run directory name.",
)

parser.add_argument(
    "--policy_mode",
    type=str,
    choices=["policy", "offset"],
    default="policy",
    help="Policy type: full policy or offset policy.",
)

parser.add_argument(
    "--randomization_mode",
    type=str,
    choices=["no_rand", "rand"],
    default="no_rand",
    help="Enable or disable domain randomization.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args



# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import pickle
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.algos_torch import model_builder
from rl_games.torch_runner import Runner

from components.networks import RMANetworkBuilder, MLPNetworkBuilder, MLPWithRNNNetworkBuilder, RMAWithTCNNetworkBuilder

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# Register custom tasks
from tasks.robomaster import RobomasterEnv, RobomasterEnvCfg
from tasks. robomaster_racetrack import RacetrackEnv, RacetrackEnvCfg, EventCfg
# from tasks.mm_ranger import MMRangerEnv, MMRangerEnvCfg 


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    # log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # ----------------------------------------------------------
    # Explicit experiment name
    # ----------------------------------------------------------

    if args_cli.experiment_name is not None:
        agent_cfg["params"]["config"]["full_experiment_name"] = (
            args_cli.experiment_name
        )

    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get(
        "full_experiment_name",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    # dump the configuration into log-directory
    params_dir = os.path.join(log_root_path, log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)

    with open(os.path.join(params_dir, "env.pkl"), "wb") as f:
        pickle.dump(env_cfg, f)

    with open(os.path.join(params_dir, "agent.pkl"), "wb") as f:
        pickle.dump(agent_cfg, f)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    if args_cli.task == "Racetrack":

        env_cfg.offset = (
            args_cli.policy_mode == "offset"
        )

        env_cfg.domain_rand = (
            args_cli.randomization_mode == "rand"
        )

        env_cfg.events = (
            EventCfg()
            if env_cfg.domain_rand
            else None
        )

        env_cfg.optimized = args_cli.optimized


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    print("\n===== DEBUG =====")
    print("seed:", env_cfg.seed)
    print("domain_rand:", env_cfg.domain_rand)
    print("events:", env_cfg.events)
    print("optimized:", env_cfg.optimized)
    print("=================\n")
    
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    print(f'ENV: {env}')

    # ---------------- PROFILER TEST ----------------
    # import torch

    # print("\n[INFO] Running environment profiler test...")

    # obs = env.reset()

    # torch.cuda.synchronize()

    # with torch.autograd.profiler.profile(use_device='cuda') as prof:
    #     for _ in range(200):
    #         actions = torch.randn(env.num_envs, env.action_space.shape[0], device=rl_device)
    #         env.step(actions)

    # torch.cuda.synchronize()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    # print(prof.key_averages(group_by_input_shape=True).table(
    #     sort_by="self_cpu_time_total",
    #     row_limit=30
    # ))

    # print("[INFO] Profiler test finished.\n")
    # ------------------------------------------------


    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    model_builder.register_network('rma', lambda **kwargs: RMANetworkBuilder())
    model_builder.register_network('mlp_net', lambda **kwargs: MLPNetworkBuilder())
    model_builder.register_network('mlp_rnn', lambda **kwargs: MLPWithRNNNetworkBuilder())
    model_builder.register_network('rma_tcn', lambda **kwargs: RMAWithTCNNetworkBuilder())

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
