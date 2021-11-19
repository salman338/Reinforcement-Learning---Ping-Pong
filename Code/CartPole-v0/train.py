import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

from plot import plot_learning_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='CartPole-v0')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env.seed(1)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)

    # Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    steps = 0
    n_episode, n_return, n_eps = [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env)
        
        while not done:
            # Get action from DQN.
            action = dqn.act(obs, True).item()

            # Act in the true environment.
            next_obs, reward, done, info = env.step(action)

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env)
                action = torch.tensor(action, device=device).long()
                reward = preprocess(reward, env=args.env)
            else:
                next_obs = preprocess(next_obs, env=args.env)
                action = torch.tensor(action, device=device).long()
                reward = preprocess(-1, env=args.env)

            # Add the transition to the replay memory.
            done_ = torch.tensor(done, device=device).int()
            memory.push(obs, action, next_obs, reward, done_)

            steps = steps + 1

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if steps % env_config["train_frequency"] == 0:
                optimize(dqn=dqn, target_dqn=target_dqn, memory=memory, optimizer=optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if steps % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            obs = next_obs

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)

            # print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                # print('Best performance so far! Saving model.')
                print(f'Best reward {mean_return} so far in episode {episode}/{env_config["n_episodes"]}')
                print(f'Saving model for epsilon: {dqn.eps}')
                torch.save(dqn, f'models/{args.env}_test01.pt')

            n_episode.append(episode)
            n_return.append(mean_return)
            n_eps.append(dqn.eps)

    fig_file = f'plots/{args.env}_test01.png'
    plot_learning_curve(n_episode, n_return, n_eps, fig_file)

    # Close environment after training is completed.
    env.close()
