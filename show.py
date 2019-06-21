import numpy as np
import torch
import gym
import argparse
import os
import pickle

import utils
import TD3
import OurDDPG
import DDPG


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in xrange(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            env.render()
            avg_reward += reward

    avg_reward /= eval_episodes

    print "---------------------------------------"
    print "Evaluation over %d episodes: %f" % (eval_episodes, avg_reward)
    print "---------------------------------------"
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v1")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_episodes", default=1, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--is_ro", action="store_true", default=False)  # Whether or not models are saved
    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print "---------------------------------------"
    print "Settings: %s" % (file_name)
    print "---------------------------------------"

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3":
        policy = \
            TD3.TD3(state_dim, action_dim, max_action, actor_lr=args.actor_lr, is_ro=args.is_ro)
    elif args.policy_name == "OurDDPG":
        policy = \
            OurDDPG.DDPG(state_dim, action_dim, max_action, actor_lr=args.actor_lr, is_ro=args.is_ro)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action)

    policy.load("TD3_Ant-v1_0", "pytorch_models")

    evaluate_policy(policy, args.eval_episodes)
