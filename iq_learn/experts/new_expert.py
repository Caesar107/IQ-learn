"""This is the runner of using PIRO to infer the reward functions and the optimal policy

"""
import multiprocessing as mp
import torch
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import policies, MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from imitation.algorithms import bc
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

import gymnasium as gym

from imitation.data import rollout

# Import the argument parser from arguments.py
from arguments import parse_args

from typing import (
    List,
)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training an expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,  # Using demo_batch_size from args
        ent_coef=arglist.ent_coef,
        learning_rate=1e-4,
        gamma=arglist.discount,
        n_epochs=20,
        n_steps=64
    )
    expert.learn(10000, progress_bar=True)

    expert.save(f"./expert_data/{arglist.env_name}")
    return expert


def sample_expert_transitions(expert: policies):
    print("Sampling expert transitions.")
    trajs = rollout.generate_trajectories(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=arglist.n_episodes_adv_fn_est),  # Using episodes from args
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(trajs)

    torch.save(transitions, f"./expert_data/transitions_{arglist.env_name}.npy")
    # torch.save(rollouts,f"./imitation/imitation_expert/rollouts_{env_name}.npy")

    return transitions


if __name__ == '__main__':

    # make environment
    #mp.set_start_method('spawn', force=True)
    arglist = parse_args()  # Using the imported parse_args function

    # Set device
    device = torch.device(arglist.device if torch.cuda.is_available() and arglist.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")

    rng = np.random.default_rng(0)

    env = make_vec_env(
        arglist.env_name,
        n_envs=arglist.n_env,
        rng=rng,
        #parallel=True,
        #max_episode_steps=50,
    )

    print(f"Environment: {arglist.env_name}")
    print(f"Environment type: {type(env)}")

    # TODO: If the environment is running for the first time (i.e., no expert data is present in the folder), please execute the following code first.
    expert = train_expert()  # uncomment to train your own expert
    
    # load expert data
    #expert = PPO.load(f"./expert_data/{arglist.env_name}")
    #transitions = torch.load(f"./expert_data/transitions_{arglist.env_name}.npy")
    transitions = sample_expert_transitions(expert)

    mean_reward, std_reward = evaluate_policy(model=expert, env=env)
    print(f"Average reward of the expert: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}")

    # @truncate the length of expert transition
    transitions = transitions[:arglist.transition_truncate_len]
    #print(transitions)

    obs = transitions.obs
    actions = transitions.acts
    infos = transitions.infos
    next_obs = transitions.next_obs
    dones = transitions.dones

    # Save current time for output directories
    time_str = time.strftime('%Y_%m_%d_%H:%M')
    output_dir = f"{arglist.save_results_dir}/expert_{arglist.env_name}_{time_str}"
    
    print(f"Results will be saved to: {output_dir}")
    
    # initiate reward_net (uncomment and complete when needed)
    env_spec = gym.spec(arglist.env_name)
    env_temp = env_spec.make()
    observation_space = env_temp.observation_space
    action_space = env_temp.action_space
    print(f"Observation space: {observation_space}, Action space: {action_space}")



