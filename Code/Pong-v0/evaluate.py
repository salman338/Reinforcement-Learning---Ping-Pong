import argparse
import random

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# parser.add_argument('--env', choices=['Pong-v0'])
parser.add_argument('--env', default='Pong-v0')
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
# parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
# parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
# parser.set_defaults(render=False)
# parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'Pong-v0': config.Pong
}

obs_stack_size = 4

def evaluate_policy(dqn, env, env_config, args, n_episodes):
    """
    Runs {n_episodes} episodes to evaluate current policy.
    """
    total_return = 0

    for i in range(n_episodes):
        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        obs_stack = torch.cat(obs_stack_size * [obs]).unsqueeze(0).to(device)

        done = False
        episode_return = 0

        while not done:
            action = dqn.act(obs_stack, exploit=True).item()

            obs, reward, done, info = env.step(action)
            obs = preprocess(obs, env=args.env).unsqueeze(0)
            obs_stack = torch.cat((obs_stack[:, 1:, ...], obs.unsqueeze(1)), dim=1).to(device)

            episode_return += reward
        
        total_return += episode_return

    return total_return / n_episodes
