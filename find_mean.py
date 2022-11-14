# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


"""
VM: cpu, cpu, mem, mem, cpu % 16, cpu % 16  (0 is full, 1 is empty)
PM: cpu, cpu, mem, mem, fragment_rate, cpu % 16, fragment_rate, cpu % 16
cpu % 16 = round(normalized_cpu * 88) % 16 / 16
fragment_rate = round(normalized_cpu * 88) % 16 / round(normalized_cpu * 88)
To rescale memory, mem * 368776
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import pandas as pd
import wandb

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import gym_reschdule_combination.envs.vm_rescheduler_env

import models
import utils
from env_patch import AsyncVectorEnv_Patch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attn", help="model architecture")
    parser.add_argument("--pretrain", action='store_true',
                        help="if toggled, we will restore pretrained weights for vm selection")
    parser.add_argument("--gym-id", type=str, default="generalizer-v1",
                        help="the id of the gym environment")
    parser.add_argument("--vm-data-size", type=str, default="M", choices=["M", "L"],
                        help="size of the dataset")
    parser.add_argument("--max-steps", type=int, default=50, help="maximum number of redeploy steps")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--normalize", action='store_true',
                        help="if toggled, we will normalize the input features")
    parser.add_argument("--track", action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--debug", action='store_true',
                        help="if toggled, this experiment will save run details")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--accum-iter", type=int, default=4,
                        help="number of iterations where gradient is accumulated before the weights are updated;"
                             " used to increase the effective batch size")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.005,  # 0.01
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1e-2,  # 1e-4
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // (args.num_minibatches * args.accum_iter))
    return args


def make_env(gym_id, seed, vm_data_size, max_steps, normalize):
    def thunk():
        env = gym.make(gym_id, seed=seed, vm_data_size=vm_data_size, max_steps=max_steps, normalize=normalize)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    num_train = 6000
    num_dev = 200
    num_test = 200
    num_envs = args.num_envs
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = AsyncVectorEnv_Patch(
        [make_env(args.gym_id, args.seed + i, args.vm_data_size, args.max_steps,
                  args.normalize) for i in range(num_envs)]
    )

    params = utils.Params(f'./experiments/pretrain/{args.model}/params.json')
    params.update('./data/params.json')

    envs.call('set_mode', mode='dev')
    pm_mean = np.zeros((num_dev, 8))
    pm_std = np.zeros((num_dev, 8))
    vm_mean = np.zeros((num_dev, 6))
    vm_std = np.zeros((num_dev, 6))

    dev_pbar = trange(0, num_dev, num_envs, desc='Dev')
    for i, file_index in enumerate(dev_pbar):
        file_ids = [num_train + file_index + env_id for env_id in range(num_envs)]
        envs.call_parse('set_current_env', env_id=file_ids)

        info = envs.reset()
        vm_mean[i * num_envs: (i+1) * num_envs] = info['vm_mean'][:, 0]
        vm_std[i * num_envs: (i+1) * num_envs] = info['vm_std'][:, 0]
        pm_mean[i * num_envs: (i+1) * num_envs] = info['pm_mean'][:, 0]
        pm_std[i * num_envs: (i+1) * num_envs] = info['pm_std'][:, 0]

    vm_all_mean = np.mean(vm_mean, axis=0)
    vm_all_std = np.mean(vm_std, axis=0)
    pm_all_mean = np.mean(pm_mean, axis=0)
    pm_all_std = np.mean(pm_std, axis=0)
    print(f'dev: vm_mean = {vm_all_mean}, vm_std = {vm_all_std}, pm_mean = {pm_all_mean}, pm_std = {pm_all_std}')
    print(f'dev frag: {pm_mean[:, 4]}')
    print(f'dev frag: {pm_std[:, 4]}')

    envs.call('set_mode', mode='test')
    pm_mean = np.zeros((num_test, 8))
    pm_std = np.zeros((num_test, 8))
    vm_mean = np.zeros((num_test, 6))
    vm_std = np.zeros((num_test, 6))

    test_pbar = trange(0, num_test, num_envs, desc='Test')
    for i, file_index in enumerate(test_pbar):
        file_ids = [num_train + num_dev + file_index + env_id for env_id in range(num_envs)]
        envs.call_parse('set_current_env', env_id=file_ids)

        info = envs.reset()
        vm_mean[i * num_envs: (i+1) * num_envs] = info['vm_mean'][:, 0]
        vm_std[i * num_envs: (i+1) * num_envs] = info['vm_std'][:, 0]
        pm_mean[i * num_envs: (i+1) * num_envs] = info['pm_mean'][:, 0]
        pm_std[i * num_envs: (i+1) * num_envs] = info['pm_std'][:, 0]

    vm_all_mean = np.mean(vm_mean, axis=0)
    vm_all_std = np.mean(vm_std, axis=0)
    pm_all_mean = np.mean(pm_mean, axis=0)
    pm_all_std = np.mean(pm_std, axis=0)
    print(f'test: vm_mean = {vm_all_mean}, vm_std = {vm_all_std}, pm_mean = {pm_all_mean}, pm_std = {pm_all_std}')
    print(f'test frag: {pm_mean[:, 4]}')
    print(f'test frag: {pm_std[:, 4]}')

    envs.call('set_mode', mode='train')
    pm_mean = np.zeros((4000, 8))
    pm_std = np.zeros((4000, 8))
    vm_mean = np.zeros((4000, 6))
    vm_std = np.zeros((4000, 6))

    train_pbar = trange(0, 4000, num_envs, desc='Train')
    for i, file_index in enumerate(train_pbar):
        file_ids = [file_index + env_id for env_id in range(num_envs)]
        envs.call_parse('set_current_env', env_id=file_ids)

        info = envs.reset()
        vm_mean[i * num_envs: (i+1) * num_envs] = info['vm_mean'][:, 0]
        vm_std[i * num_envs: (i+1) * num_envs] = info['vm_std'][:, 0]
        pm_mean[i * num_envs: (i+1) * num_envs] = info['pm_mean'][:, 0]
        pm_std[i * num_envs: (i+1) * num_envs] = info['pm_std'][:, 0]

    vm_all_mean = np.mean(vm_mean, axis=0)
    vm_all_std = np.mean(vm_std, axis=0)
    pm_all_mean = np.mean(pm_mean, axis=0)
    pm_all_std = np.mean(pm_std, axis=0)
    print(f'train: vm_mean = {vm_all_mean}, vm_std = {vm_all_std}, pm_mean = {pm_all_mean}, pm_std = {pm_all_std}')

    vm_all_mean1 = np.mean(vm_mean[-200:], axis=0)
    vm_all_std1 = np.mean(vm_std[-200:], axis=0)
    pm_all_mean1 = np.mean(pm_mean[-200:], axis=0)
    pm_all_std1 = np.mean(pm_std[-200:], axis=0)
    print(f'train1: vm_mean = {vm_all_mean1}, vm_std = {vm_all_std1}, pm_mean = {pm_all_mean1}, pm_std = {pm_all_std1}')
    print(f'train1 frag: {pm_mean[-200:, 4]}')
    print(f'train1 frag: {pm_std[-200:, 4]}')

    vm_all_mean2 = np.mean(vm_mean[-400:-200], axis=0)
    vm_all_std2 = np.mean(vm_std[-400:-200], axis=0)
    pm_all_mean2 = np.mean(pm_mean[-400:-200], axis=0)
    pm_all_std2 = np.mean(pm_std[-400:-200], axis=0)
    print(f'train2: vm_mean = {vm_all_mean2}, vm_std = {vm_all_std2}, pm_mean = {pm_all_mean2}, pm_std = {pm_all_std2}')
    print(f'train2 frag: {pm_mean[-400:-200, 4]}')
    print(f'train2 frag: {pm_std[-400:-200, 4]}')

    envs.close()
