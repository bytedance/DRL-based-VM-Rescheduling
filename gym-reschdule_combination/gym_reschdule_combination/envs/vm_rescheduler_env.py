# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


import gym
from gym.utils import seeding
import os
import json
import numpy as np
import random
from gym import spaces
from gym.utils import seeding

from collections import defaultdict


# return the vm request
# @timer("数据加载")
def parse_input(input_stream):
    # print(input_stream)
    if os.path.isfile(input_stream):
        with open(input_stream, 'r', encoding='utf-8') as f:
            input_stream = json.load(f)

    # extract input data
    clusters = input_stream['cluster_list']
    vm_types = input_stream['vm_type_list']
    requirements = input_stream['general_requirement']

    scheduler = VirtualMachineScheduler(clusters, vm_types, requirements)

    return scheduler


class VM_generlizer_v0(gym.Env):

    def __init__(self, seed, vm_data_size, max_steps, normalize=True):

        self.action_space = gym.spaces.MultiDiscrete([2089, 279])
        self.MAX_STEPS = max_steps
        self.vm_data_size = vm_data_size
        self.normalize = normalize
        self.n_pms = 279
        self.n_vms = 2089
        self.current_vms = self.n_vms
        self.mask = True
        self._mode = "train"
        self.train_range = 4000
        self._current_env = -1
        self.observation_space = spaces.Dict({
            "pm_info": spaces.Box(0, 1, shape=(self.n_pms, 8)),
            "vm_info": spaces.Box(0, 1, shape=(self.n_vms, 14)),
            "num_steps": spaces.Box(0, 1, shape=(1, 1)),
            "num_vms": spaces.Discrete(self.n_vms),
        })
        self.pm_mean = np.array([0.0631, 0.0631, 0.0308, 0.0308, 0.3715, 0.1209, 0.3715, 0.1209])
        self.pm_std = np.array([0.1014, 0.1014, 0.0496, 0.0496, 0.4672, 0.1646, 0.4672, 0.1646])
        self.vm_mean = np.array([0.0631, 0.0631, 0.0308, 0.0308, 0.1209, 0.1209])
        self.vm_std = np.array([0.1014, 0.1014, 0.0496, 0.0496, 0.1646, 0.1646])
        random.seed(seed)

    def set_mode(self, mode):
        self._mode = mode

    def set_current_env(self, env_id):
        self._current_env = env_id

    # used_pm_status, all_pm_free_cpu, all_pm_free_mem, fragment_rate_numa0, self.fragment_mode_16_numa0
    # fragment_rate_numa1, self.fragment_mode_16_numa1
    def gather_pm_features(self, pm):
        numa0_free_cpu, numa1_free_cpu = pm.numas[0].free_cpu, pm.numas[1].free_cpu
        return [min(len(pm.vms), 1), numa0_free_cpu, numa1_free_cpu, pm.numas[0].free_mem, pm.numas[1].free_mem,
                (numa0_free_cpu % 16) / numa0_free_cpu if numa0_free_cpu else 0, (numa0_free_cpu % 16) / 16,
                (numa1_free_cpu % 16) / numa1_free_cpu if numa1_free_cpu else 0, (numa1_free_cpu % 16) / 16]

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        start_id is used to train(random value between(0, len(vms))) and test(0).

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        if self._mode == "train":
            self.scheduler = parse_input(f"data/reset_dataset/train/"
                                         f"reset_vm_pm{random.randrange(self.train_range)}.json")
        else:
            self.scheduler = parse_input(f"data/reset_dataset/{self._mode}/reset_vm_pm{self._current_env}.json")

        self.request = self.scheduler.vm_types
        self.clusters = self.scheduler.clusters
        self.pms = self.scheduler.active_pms
        self.vms = self.scheduler.migratable_vms

        # add index to all active pms.
        for i in range(len(self.pms)):
            self.pms[i].index = i

        # add index to all migratable vms.
        for i in range(len(self.vms)):
            self.vms[i].index = i

        pm_all_info = np.array(list(map(self.gather_pm_features, self.pms)))
        self.used_pm_status = pm_all_info[:, 0:1]
        self.all_pm_free_cpu = pm_all_info[:, 1:3]
        self.all_pm_free_mem = pm_all_info[:, 3:5]
        self.fragment_rate_numa0 = pm_all_info[:, 5:6]
        self.fragment_mode_16_numa0 = pm_all_info[:, 6:7]
        self.fragment_rate_numa1 = pm_all_info[:, 7:8]
        self.fragment_mode_16_numa1 = pm_all_info[:, 8:9]

        self.all_pm_cpu = np.sum(self.all_pm_free_cpu)
        self.all_pm_mem = np.sum(self.all_pm_free_mem)

        # get the request info
        self.vm_cpu_numa0 = []
        self.vm_cpu_numa1 = []
        self.vm_mem_numa0 = []
        self.vm_mem_numa1 = []
        self.vm_request_cpu = []
        self.vm_request_mem = []

        self.vm_frag_mode16_numa0 = []
        self.vm_frag_mode16_numa1 = []

        for vm in self.vms:
            self.vm_request_cpu.append(vm.cpu)
            self.vm_request_mem.append(vm.mem)
            if len(vm.deploy_numa) == 1:
                numa_cpu = vm.cpu
                if vm.deploy_numa[0] == 1:
                    self.vm_cpu_numa0.append(0)
                    self.vm_cpu_numa1.append(numa_cpu)
                    self.vm_mem_numa0.append(0)
                    self.vm_mem_numa1.append(vm.mem)

                    self.vm_frag_mode16_numa0.append(0)
                    self.vm_frag_mode16_numa1.append((numa_cpu % 16) / 16)
                else:
                    self.vm_cpu_numa0.append(numa_cpu)
                    self.vm_cpu_numa1.append(0)
                    self.vm_mem_numa0.append(vm.mem)
                    self.vm_mem_numa1.append(0)

                    self.vm_frag_mode16_numa0.append((numa_cpu % 16) / 16)
                    self.vm_frag_mode16_numa1.append(0)

            else:
                numa_cpu = int(vm.cpu / 2)
                numa_cpu_mod16 = (numa_cpu % 16) / 16
                numa_mem = int(vm.mem / 2)
                self.vm_cpu_numa0.append(numa_cpu)
                self.vm_cpu_numa1.append(numa_cpu)
                self.vm_mem_numa0.append(numa_mem)
                self.vm_mem_numa1.append(numa_mem)

                self.vm_frag_mode16_numa0.append(numa_cpu_mod16)
                self.vm_frag_mode16_numa1.append(numa_cpu_mod16)

        self.vm_cpu_numa0 = np.array(self.vm_cpu_numa0) / 88
        self.vm_cpu_numa1 = np.array(self.vm_cpu_numa1) / 88
        self.vm_mem_numa0 = np.array(self.vm_mem_numa0) / 368776
        self.vm_mem_numa1 = np.array(self.vm_mem_numa1) / 368776
        self.vm_frag_mode16_numa0 = np.array(self.vm_frag_mode16_numa0)
        self.vm_frag_mode16_numa1 = np.array(self.vm_frag_mode16_numa1)

        n = len(self.vms)
        self.demands = np.vstack([np.arange(n) / n, self.vm_cpu_numa0, self.vm_cpu_numa1,
                                  self.vm_mem_numa0, self.vm_mem_numa1,
                                  self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T
        self.current_step = 0
        self.get_obs_()

        self.reward = 0
        self.done = False
        self.info = {}
        # self.init_frag_rate = self.scheduler.get_fragment_rate()

        return self.state

    def step(self, action):  # @timer("schedule one vm")
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        done = False

        assert self.action_space.contains(action)
        pm_state = self.current_pm_state
        demand = self.demands[action[0]][1:5]
        real = self.pms[action[1]].can_place(self.vms[action[0]])

        if real != self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4), np.round(demand, 4)):
            import pdb
            pdb.set_trace()

        vm = self.vms[action[0]]  # 要迁移的虚机
        pm_dest = self.pms[action[1]]  # 迁移的目的地物理机

        pm_source = vm.pm  # 迁移的虚机的源物理机
        pm_source_id = vm.pm.index  # 迁移的虚机的源物理机id

        assert real == self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4),
                                           np.round(demand, 4)), \
            f"real = {real}, can_pm_meet_vm = {self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4), np.round(demand, 4))}"

        if pm_source_id == action[1]:
            reward = 0
        elif not real:
            print(f'real: {real}, is_source = {pm_source_id == action[1]}')
            pm_mask = self.get_pm_mask(int(action[0]))
            print('all_mask: ', pm_mask)
            print('pm_mask: ', pm_mask[int(action[1])])
            print('real: ', self.pms[action[1]].can_place(self.vms[action[0]]))
            raise ValueError('PM action is not fully masked. Improper action selected!')
        else:
            frag_rate_before_mig = self.get_fragment_rate_reward([pm_source, pm_dest])
            self.pm_add_vm(pm_state[action[1], :], demand, pm_state[pm_source_id, :], pm_dest, vm, pm_source,
                           self.demands)

            frag_rate_after_mig = self.get_fragment_rate_reward([pm_source, pm_dest])
            reward = (sum(frag_rate_before_mig) - sum(frag_rate_after_mig)) / 4

        self.current_step += 1

        if self.current_step >= self.MAX_STEPS:
            done = True
        self.get_obs(pm_state)

        return self.state, reward, done, {
            "fragment_rate": self.scheduler.get_fragment_rate()}  # "init_frag_rate": self.init_frag_rate

    def can_pm_meet_vm(self, pm, vm):

        if vm[0] == 0 and vm[2] == 0:  # numa 1, cpu and mem
            if (pm[1] < vm[1] or pm[3] < vm[3]) and (pm[0] < vm[1] or pm[2] < vm[3]):
                return False
        elif vm[1] == 0 and vm[3] == 0:  # numa 0, cpu and mem
            if (pm[0] < vm[0] or pm[2] < vm[2]) and (pm[1] < vm[0] or pm[3] < vm[2]):
                return False
        elif any(pm - vm < 0):  # double numa
            return False

        return True

    def pm_add_vm(self, pm_norm_dest, vm_norm, pm_norm_source, pm_dest, vm, pm_source, demands):
        if vm.double_numa:

            pm_norm_source[:4] += vm_norm
            pm_source.release_a_vm(vm, deploy_numa=[0, 1])

            pm_norm_source[4] = (pm_source.numas[0].free_cpu % 16) / pm_source.numas[0].free_cpu
            pm_norm_source[5] = (pm_source.numas[0].free_cpu % 16) / 16
            pm_norm_source[6] = (pm_source.numas[1].free_cpu % 16) / pm_source.numas[1].free_cpu
            pm_norm_source[7] = (pm_source.numas[1].free_cpu % 16) / 16

            pm_norm_dest[:4] -= vm_norm
            pm_dest.add_a_vm(vm, deploy_numa=[0, 1])

            if pm_dest.numas[0].free_cpu == 0:
                pm_norm_dest[4] = 0
                pm_norm_dest[5] = 0
            elif pm_dest.numas[0].free_cpu != 0:
                pm_norm_dest[4] = (pm_dest.numas[0].free_cpu % 16) / pm_dest.numas[0].free_cpu
                pm_norm_dest[5] = (pm_dest.numas[0].free_cpu % 16) / 16

            if pm_dest.numas[1].free_cpu == 0:
                pm_norm_dest[6] = 0
                pm_norm_dest[7] = 0
            elif pm_dest.numas[1].free_cpu != 0:
                pm_norm_dest[6] = (pm_dest.numas[1].free_cpu % 16) / pm_dest.numas[1].free_cpu
                pm_norm_dest[7] = (pm_dest.numas[1].free_cpu % 16) / 16

        else:

            if pm_dest.numas[0].can_place(vm):
                deploy_numa = 0
                demands[vm.index][1:] = np.array([vm.cpu / 88, 0, vm.mem / 368776, 0, (vm.cpu % 16) / 16, 0])
            elif pm_dest.numas[1].can_place(vm):
                deploy_numa = 1
                demands[vm.index][1:] = np.array([0, vm.cpu / 88, 0, vm.mem / 368776, 0, (vm.cpu % 16) / 16])
            else:
                raise ValueError(f'Cannot fit on both numa! VM: {vm_norm}, PM_dest: {pm_norm_dest}')

            current_numa = vm.deploy_numa[0]
            pm_norm_source[current_numa] += vm_norm[deploy_numa]
            pm_norm_source[current_numa + 2] += vm_norm[deploy_numa + 2]

            # update pm_source
            pm_source.release_a_vm(vm)
            pm_norm_source[current_numa + 4] = (pm_source.numas[current_numa].free_cpu % 16) / pm_source.numas[
                current_numa].free_cpu
            pm_norm_source[current_numa + 5] = (pm_source.numas[current_numa].free_cpu % 16) / 16

            # update pm_dest
            pm_norm_dest[deploy_numa] -= vm_norm[deploy_numa]
            pm_norm_dest[deploy_numa + 2] -= vm_norm[deploy_numa + 2]

            pm_dest.add_a_vm(vm, deploy_numa=[deploy_numa])
            pm_norm_dest[deploy_numa + 4] = 0 if pm_dest.numas[deploy_numa].free_cpu == 0 else (pm_dest.numas[
                                                                                                    deploy_numa].free_cpu % 16) / 16
            pm_norm_dest[deploy_numa + 5] = (pm_dest.numas[deploy_numa].free_cpu % 16) / 16

        pm_norm_dest[pm_norm_dest < 0.000001] = 0

    def get_pm_mask(self, vm_id):
        pm_mask = np.zeros(len(self.pms))
        for index, pm in enumerate(self.pms):
            if not pm.can_place(self.vms[vm_id]):
                pm_mask[index] = 1

        pm_mask[self.vms[vm_id].pm.index] = 0
        return pm_mask

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def get_obs_(self):  # used for reset

        # import pdb
        # pdb.set_trace()
        self.current_pm_state = np.hstack((self.all_pm_free_cpu / 88, self.all_pm_free_mem / 368776,
                                           self.fragment_rate_numa0, self.fragment_mode_16_numa0,
                                           self.fragment_rate_numa1, self.fragment_mode_16_numa1))

        # construct vm info
        vm_state1 = np.vstack(
            [self.vm_cpu_numa0, self.vm_cpu_numa1, self.vm_mem_numa0,
             self.vm_mem_numa1, self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T
        if self.normalize:
            self.state = {'pm_info': (self.current_pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.hstack([(vm_state1 - self.vm_mean) / self.vm_std,
                                                (np.array([self.current_pm_state[vm.pm.index] for vm in
                                                           self.vms]) - self.pm_mean) / self.pm_std]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }
        else:
            self.state = {'pm_info': self.current_pm_state,
                          'vm_info': np.hstack([vm_state1, np.array([self.current_pm_state[vm.pm.index] for vm in
                                                                     self.vms])]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }

    def get_obs(self, pm_state):
        if self.normalize:
            self.state = {'pm_info': (pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.hstack([(self.demands[:, 1:] - self.vm_mean) / self.vm_std,
                                                (np.array([pm_state[vm.pm.index] for vm in
                                                           self.vms]) - self.pm_mean) / self.pm_std]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }
        else:
            self.state = {'pm_info': pm_state,
                          'vm_info': np.hstack(
                              [self.demands[:, 1:], np.array([pm_state[vm.pm.index] for vm in self.vms])]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }

    def get_fragment_rate_reward(self, source_dest_pms):
        fragment_rate = []
        for pm in source_dest_pms:
            fragment_rate.append((pm.numas[0].free_cpu % 16) / 32 + (pm.numas[1].free_cpu % 16) / 32)
        return fragment_rate


class VM_generlizer_v1(VM_generlizer_v0):

    def __init__(self, seed, vm_data_size, max_steps, normalize=True):
        super(VM_generlizer_v1, self).__init__(seed=seed, vm_data_size=vm_data_size, max_steps=max_steps,
                                               normalize=normalize)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        start_id is used to train(random value between(0, len(vms))) and test(0).

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        if self._mode == "train":
            self.scheduler = parse_input(f"data/flex_vm_dataset/{self.vm_data_size}/train/"
                                         f"flex_vm_{random.randrange(self.train_range)}.json")
        else:
            self.scheduler = parse_input(f"data/flex_vm_dataset/{self.vm_data_size}/{self._mode}/"
                                         f"flex_vm_{self._current_env}.json")
        self.request = self.scheduler.vm_types
        self.clusters = self.scheduler.clusters
        self.pms = self.scheduler.active_pms
        self.vms = self.scheduler.migratable_vms
        self.current_vms = len(self.vms)

        # add index to all active pms.
        for i in range(len(self.pms)):
            self.pms[i].index = i

        # add index to all migratable vms.
        for i in range(len(self.vms)):
            self.vms[i].index = i

        pm_all_info = np.array(list(map(self.gather_pm_features, self.pms)))
        self.used_pm_status = pm_all_info[:, 0:1]
        self.all_pm_free_cpu = pm_all_info[:, 1:3]
        self.all_pm_free_mem = pm_all_info[:, 3:5]
        self.fragment_rate_numa0 = pm_all_info[:, 5:6]
        self.fragment_mode_16_numa0 = pm_all_info[:, 6:7]
        self.fragment_rate_numa1 = pm_all_info[:, 7:8]
        self.fragment_mode_16_numa1 = pm_all_info[:, 8:9]

        self.all_pm_cpu = np.sum(self.all_pm_free_cpu)
        self.all_pm_mem = np.sum(self.all_pm_free_mem)

        # get the request info
        self.vm_cpu_numa0 = []
        self.vm_cpu_numa1 = []
        self.vm_mem_numa0 = []
        self.vm_mem_numa1 = []
        self.vm_request_cpu = []
        self.vm_request_mem = []

        self.vm_frag_mode16_numa0 = []
        self.vm_frag_mode16_numa1 = []

        for vm in self.vms:
            self.vm_request_cpu.append(vm.cpu)
            self.vm_request_mem.append(vm.mem)
            if len(vm.deploy_numa) == 1:
                numa_cpu = vm.cpu
                if vm.deploy_numa[0] == 1:
                    self.vm_cpu_numa0.append(0)
                    self.vm_cpu_numa1.append(numa_cpu)
                    self.vm_mem_numa0.append(0)
                    self.vm_mem_numa1.append(vm.mem)

                    self.vm_frag_mode16_numa0.append(0)
                    self.vm_frag_mode16_numa1.append((numa_cpu % 16) / 16)
                else:
                    self.vm_cpu_numa0.append(numa_cpu)
                    self.vm_cpu_numa1.append(0)
                    self.vm_mem_numa0.append(vm.mem)
                    self.vm_mem_numa1.append(0)

                    self.vm_frag_mode16_numa0.append((numa_cpu % 16) / 16)
                    self.vm_frag_mode16_numa1.append(0)

            else:
                numa_cpu = int(vm.cpu / 2)
                numa_cpu_mod16 = (numa_cpu % 16) / 16
                numa_mem = int(vm.mem / 2)
                self.vm_cpu_numa0.append(numa_cpu)
                self.vm_cpu_numa1.append(numa_cpu)
                self.vm_mem_numa0.append(numa_mem)
                self.vm_mem_numa1.append(numa_mem)

                self.vm_frag_mode16_numa0.append(numa_cpu_mod16)
                self.vm_frag_mode16_numa1.append(numa_cpu_mod16)

        self.vm_cpu_numa0 = np.array(self.vm_cpu_numa0) / 88
        self.vm_cpu_numa1 = np.array(self.vm_cpu_numa1) / 88
        self.vm_mem_numa0 = np.array(self.vm_mem_numa0) / 368776
        self.vm_mem_numa1 = np.array(self.vm_mem_numa1) / 368776
        self.vm_frag_mode16_numa0 = np.array(self.vm_frag_mode16_numa0)
        self.vm_frag_mode16_numa1 = np.array(self.vm_frag_mode16_numa1)

        n = len(self.vms)
        self.demands_before = np.vstack([np.arange(n) / n, self.vm_cpu_numa0, self.vm_cpu_numa1,
                                         self.vm_mem_numa0, self.vm_mem_numa1,
                                         self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T

        # add 0 to the rest
        self.vm_placeholder = np.zeros((self.n_vms - self.current_vms, 14))
        self.demands = np.vstack((self.demands_before, self.vm_placeholder[:, :7]))

        self.current_step = 0
        self.get_obs_()
        self.reward = 0
        self.done = False
        self.info = {}
        # self.init_frag_rate = self.scheduler.get_fragment_rate()
        return self.state

    def get_obs_(self):

        # import pdb
        # pdb.set_trace()
        self.current_pm_state = np.hstack((self.all_pm_free_cpu / 88, self.all_pm_free_mem / 368776,
                                           self.fragment_rate_numa0, self.fragment_mode_16_numa0,
                                           self.fragment_rate_numa1, self.fragment_mode_16_numa1))

        # construct vm info
        vm_state1 = np.vstack(
            [self.vm_cpu_numa0, self.vm_cpu_numa1, self.vm_mem_numa0,
             self.vm_mem_numa1, self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T

        # add 0 to the rest
        if self.normalize:
            vm_info_before = np.hstack([(vm_state1 - self.vm_mean) / self.vm_std,
                                        (np.array([self.current_pm_state[vm.pm.index] for vm in
                                                   self.vms]) - self.pm_mean) / self.pm_std])
            self.state = {'pm_info': (self.current_pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.vstack((vm_info_before, self.vm_placeholder)),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }
        else:
            vm_info_before = np.hstack([vm_state1, np.array([self.current_pm_state[vm.pm.index] for vm in self.vms])])
            self.state = {'pm_info': self.current_pm_state,
                          'vm_info': np.vstack((vm_info_before, self.vm_placeholder)),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }

    def get_obs(self, pm_state):

        # add 0 to the rest
        if self.normalize:
            vm_info_before = np.hstack([(self.demands_before[:, 1:] - self.vm_mean) / self.vm_std,
                                        (np.array([pm_state[vm.pm.index] for vm in
                                                   self.vms]) - self.pm_mean) / self.pm_std])
            self.state = {'pm_info': (pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.vstack((vm_info_before, self.vm_placeholder)),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }
        else:
            vm_info_before = np.hstack(
                [self.demands_before[:, 1:], np.array([pm_state[vm.pm.index] for vm in self.vms])])
            self.state = {'pm_info': pm_state,
                          'vm_info': np.vstack((vm_info_before, self.vm_placeholder)),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms
                          }


class VM_graph_v1(VM_generlizer_v1):
    def __init__(self, seed, vm_data_size, max_steps, normalize=True):
        super(VM_graph_v1, self).__init__(seed=seed, vm_data_size=vm_data_size, max_steps=max_steps,
                                          normalize=normalize)
        self.observation_space = spaces.Dict({
            "pm_info": spaces.Box(0, 1, shape=(self.n_pms, 8)),
            "vm_info": spaces.Box(0, 1, shape=(self.n_vms, 6)),
            "num_steps": spaces.Box(0, 1, shape=(1, 1)),
            "num_vms": spaces.Discrete(self.n_vms),
            "edges": spaces.Box(0, self.n_vms + self.n_pms, shape=(self.n_vms, 2)),
        })

    def reset(self):
        if self._mode == "train":
            self.scheduler = parse_input(f"data/flex_vm_dataset/{self.vm_data_size}/train/"
                                         f"flex_vm_{random.randrange(self.train_range)}.json")
        else:
            self.scheduler = parse_input(f"data/flex_vm_dataset/{self.vm_data_size}/{self._mode}/"
                                         f"flex_vm_{self._current_env}.json")

        self.request = self.scheduler.vm_types
        self.clusters = self.scheduler.clusters
        self.pms = self.scheduler.active_pms
        self.vms = self.scheduler.migratable_vms
        self.current_vms = len(self.vms)

        # add index to all active pms.
        for i in range(len(self.pms)):
            self.pms[i].index = i

        # add index to all migratable vms.
        for i in range(len(self.vms)):
            self.vms[i].index = i

        pm_all_info = np.array(list(map(self.gather_pm_features, self.pms)))
        self.used_pm_status = pm_all_info[:, 0:1]
        self.all_pm_free_cpu = pm_all_info[:, 1:3]
        self.all_pm_free_mem = pm_all_info[:, 3:5]
        self.fragment_rate_numa0 = pm_all_info[:, 5:6]
        self.fragment_mode_16_numa0 = pm_all_info[:, 6:7]
        self.fragment_rate_numa1 = pm_all_info[:, 7:8]
        self.fragment_mode_16_numa1 = pm_all_info[:, 8:9]

        self.all_pm_cpu = np.sum(self.all_pm_free_cpu)
        self.all_pm_mem = np.sum(self.all_pm_free_mem)

        # get the request info
        self.vm_cpu_numa0 = []
        self.vm_cpu_numa1 = []
        self.vm_mem_numa0 = []
        self.vm_mem_numa1 = []
        self.vm_request_cpu = []
        self.vm_request_mem = []

        self.vm_frag_mode16_numa0 = []
        self.vm_frag_mode16_numa1 = []

        for vm in self.vms:
            self.vm_request_cpu.append(vm.cpu)
            self.vm_request_mem.append(vm.mem)
            if len(vm.deploy_numa) == 1:
                numa_cpu = vm.cpu
                if vm.deploy_numa[0] == 1:
                    self.vm_cpu_numa0.append(0)
                    self.vm_cpu_numa1.append(numa_cpu)
                    self.vm_mem_numa0.append(0)
                    self.vm_mem_numa1.append(vm.mem)

                    self.vm_frag_mode16_numa0.append(0)
                    self.vm_frag_mode16_numa1.append((numa_cpu % 16) / 16)
                else:
                    self.vm_cpu_numa0.append(numa_cpu)
                    self.vm_cpu_numa1.append(0)
                    self.vm_mem_numa0.append(vm.mem)
                    self.vm_mem_numa1.append(0)

                    self.vm_frag_mode16_numa0.append((numa_cpu % 16) / 16)
                    self.vm_frag_mode16_numa1.append(0)

            else:
                numa_cpu = int(vm.cpu / 2)
                numa_cpu_mod16 = (numa_cpu % 16) / 16
                numa_mem = int(vm.mem / 2)
                self.vm_cpu_numa0.append(numa_cpu)
                self.vm_cpu_numa1.append(numa_cpu)
                self.vm_mem_numa0.append(numa_mem)
                self.vm_mem_numa1.append(numa_mem)

                self.vm_frag_mode16_numa0.append(numa_cpu_mod16)
                self.vm_frag_mode16_numa1.append(numa_cpu_mod16)

        self.vm_cpu_numa0 = np.array(self.vm_cpu_numa0) / 88
        self.vm_cpu_numa1 = np.array(self.vm_cpu_numa1) / 88
        self.vm_mem_numa0 = np.array(self.vm_mem_numa0) / 368776
        self.vm_mem_numa1 = np.array(self.vm_mem_numa1) / 368776
        self.vm_frag_mode16_numa0 = np.array(self.vm_frag_mode16_numa0)
        self.vm_frag_mode16_numa1 = np.array(self.vm_frag_mode16_numa1)

        n = len(self.vms)
        self.demands_before = np.vstack([np.arange(n) / n, self.vm_cpu_numa0, self.vm_cpu_numa1,
                                         self.vm_mem_numa0, self.vm_mem_numa1,
                                         self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T
        # add 0 to the rest
        self.vm_placeholder = np.zeros((self.n_vms - self.current_vms, 7))
        self.demands = np.vstack((self.demands_before, self.vm_placeholder))

        self.current_step = 0
        self.edges = np.zeros((self.n_vms, 2))
        for vm_id, vm in enumerate(self.vms):
            pm_id = vm.pm.index
            self.edges[vm_id, 0] = vm_id + self.n_pms
            self.edges[vm_id, 1] = pm_id
        self.get_obs_()
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state

    def step(self, action):
        done = False

        assert self.action_space.contains(action)
        pm_state = self.current_pm_state
        demand = self.demands[action[0]][1:5]
        real = self.pms[action[1]].can_place(self.vms[action[0]])

        if real != self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4), np.round(demand, 4)):
            import pdb
            pdb.set_trace()

        vm = self.vms[action[0]]  # 要迁移的虚机
        pm_dest = self.pms[action[1]]  # 迁移的目的地物理机

        pm_source = vm.pm  # 迁移的虚机的源物理机
        pm_source_id = vm.pm.index  # 迁移的虚机的源物理机id

        assert real == self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4),
                                           np.round(demand, 4)), \
            f"real = {real}, can_pm_meet_vm = {self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4), np.round(demand, 4))}"

        if pm_source_id == action[1]:
            reward = 0
        elif not real:
            print(f'real: {real}, is_source = {pm_source_id == action[1]}')
            pm_mask = self.get_pm_mask(int(action[0]))
            print('all_mask: ', pm_mask)
            print('pm_mask: ', pm_mask[int(action[1])])
            print('real: ', self.pms[action[1]].can_place(self.vms[action[0]]))
            raise ValueError('PM action is not fully masked. Improper action selected!')
        else:
            frag_rate_before_mig = self.get_fragment_rate_reward([pm_source, pm_dest])
            self.pm_add_vm(pm_state[action[1], :], demand, pm_state[pm_source_id, :], pm_dest, vm, pm_source,
                           self.demands)

            frag_rate_after_mig = self.get_fragment_rate_reward([pm_source, pm_dest])
            reward = (sum(frag_rate_before_mig) - sum(frag_rate_after_mig)) / 4

        self.current_step += 1

        if self.current_step >= self.MAX_STEPS:
            done = True

        self.edges[action[0], 0] = action[0] + self.n_pms
        self.edges[action[0], 1] = action[1]
        self.get_obs(pm_state)

        return self.state, reward, done, {
            "fragment_rate": self.scheduler.get_fragment_rate()}  # "init_frag_rate": self.init_frag_rate

    def get_obs_(self):
        self.current_pm_state = np.hstack((self.all_pm_free_cpu / 88, self.all_pm_free_mem / 368776,
                                           self.fragment_rate_numa0, self.fragment_mode_16_numa0,
                                           self.fragment_rate_numa1, self.fragment_mode_16_numa1))

        # construct vm info
        vm_state1 = np.vstack(
            [self.vm_cpu_numa0, self.vm_cpu_numa1, self.vm_mem_numa0,
             self.vm_mem_numa1, self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T
        if self.normalize:
            self.state = {'pm_info': (self.current_pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.vstack(((vm_state1 - self.vm_mean) / self.vm_std, self.vm_placeholder[:, 1:])),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }
        else:
            self.state = {'pm_info': self.current_pm_state,
                          'vm_info': np.vstack((vm_state1, self.vm_placeholder[:, 1:])),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }

    def get_obs(self, pm_state):
        if self.normalize:
            self.state = {'pm_info': (pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.vstack([(self.demands_before[:, 1:] - self.vm_mean) / self.vm_std,
                                                self.vm_placeholder[:, 1:]]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }
        else:
            self.state = {'pm_info': pm_state,
                          'vm_info': np.vstack([self.demands_before[:, 1:], self.vm_placeholder[:, 1:]]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }


class VM_graph_v2(VM_generlizer_v1):
    def __init__(self, seed, vm_data_size, max_steps, normalize=True):
        super(VM_graph_v2, self).__init__(seed=seed, vm_data_size=vm_data_size, max_steps=max_steps,
                                          normalize=normalize)
        self.observation_space = spaces.Dict({
            "pm_info": spaces.Box(0, 1, shape=(self.n_pms, 8)),
            "vm_info": spaces.Box(0, 1, shape=(self.n_vms, 6)),
            "num_steps": spaces.Box(0, 1, shape=(1, 1)),
            "num_vms": spaces.Discrete(self.n_vms),
            "edges": spaces.Box(0, self.n_vms + self.n_pms, shape=(self.n_vms + self.n_pms, 1)),
        })

    def reset(self):
        if self._mode == "train":
            self.scheduler = parse_input(f"data/flex_vm_dataset/{self.vm_data_size}/train/"
                                         f"flex_vm_{random.randrange(self.train_range)}.json")
        else:
            self.scheduler = parse_input(f"data/flex_vm_dataset/{self.vm_data_size}/{self._mode}/"
                                         f"flex_vm_{self._current_env}.json")

        self.request = self.scheduler.vm_types
        self.clusters = self.scheduler.clusters
        self.pms = self.scheduler.active_pms
        self.vms = self.scheduler.migratable_vms
        self.current_vms = len(self.vms)

        # add index to all active pms.
        for i in range(len(self.pms)):
            self.pms[i].index = i

        # add index to all migratable vms.
        for i in range(len(self.vms)):
            self.vms[i].index = i

        pm_all_info = np.array(list(map(self.gather_pm_features, self.pms)))
        self.used_pm_status = pm_all_info[:, 0:1]
        self.all_pm_free_cpu = pm_all_info[:, 1:3]
        self.all_pm_free_mem = pm_all_info[:, 3:5]
        self.fragment_rate_numa0 = pm_all_info[:, 5:6]
        self.fragment_mode_16_numa0 = pm_all_info[:, 6:7]
        self.fragment_rate_numa1 = pm_all_info[:, 7:8]
        self.fragment_mode_16_numa1 = pm_all_info[:, 8:9]

        self.all_pm_cpu = np.sum(self.all_pm_free_cpu)
        self.all_pm_mem = np.sum(self.all_pm_free_mem)

        # get the request info
        self.vm_cpu_numa0 = []
        self.vm_cpu_numa1 = []
        self.vm_mem_numa0 = []
        self.vm_mem_numa1 = []
        self.vm_request_cpu = []
        self.vm_request_mem = []

        self.vm_frag_mode16_numa0 = []
        self.vm_frag_mode16_numa1 = []

        for vm in self.vms:
            self.vm_request_cpu.append(vm.cpu)
            self.vm_request_mem.append(vm.mem)
            if len(vm.deploy_numa) == 1:
                numa_cpu = vm.cpu
                if vm.deploy_numa[0] == 1:
                    self.vm_cpu_numa0.append(0)
                    self.vm_cpu_numa1.append(numa_cpu)
                    self.vm_mem_numa0.append(0)
                    self.vm_mem_numa1.append(vm.mem)

                    self.vm_frag_mode16_numa0.append(0)
                    self.vm_frag_mode16_numa1.append((numa_cpu % 16) / 16)
                else:
                    self.vm_cpu_numa0.append(numa_cpu)
                    self.vm_cpu_numa1.append(0)
                    self.vm_mem_numa0.append(vm.mem)
                    self.vm_mem_numa1.append(0)

                    self.vm_frag_mode16_numa0.append((numa_cpu % 16) / 16)
                    self.vm_frag_mode16_numa1.append(0)

            else:
                numa_cpu = int(vm.cpu / 2)
                numa_cpu_mod16 = (numa_cpu % 16) / 16
                numa_mem = int(vm.mem / 2)
                self.vm_cpu_numa0.append(numa_cpu)
                self.vm_cpu_numa1.append(numa_cpu)
                self.vm_mem_numa0.append(numa_mem)
                self.vm_mem_numa1.append(numa_mem)

                self.vm_frag_mode16_numa0.append(numa_cpu_mod16)
                self.vm_frag_mode16_numa1.append(numa_cpu_mod16)

        self.vm_cpu_numa0 = np.array(self.vm_cpu_numa0) / 88
        self.vm_cpu_numa1 = np.array(self.vm_cpu_numa1) / 88
        self.vm_mem_numa0 = np.array(self.vm_mem_numa0) / 368776
        self.vm_mem_numa1 = np.array(self.vm_mem_numa1) / 368776
        self.vm_frag_mode16_numa0 = np.array(self.vm_frag_mode16_numa0)
        self.vm_frag_mode16_numa1 = np.array(self.vm_frag_mode16_numa1)

        n = len(self.vms)
        self.demands_before = np.vstack([np.arange(n) / n, self.vm_cpu_numa0, self.vm_cpu_numa1,
                                         self.vm_mem_numa0, self.vm_mem_numa1,
                                         self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T
        # add 0 to the rest
        self.vm_placeholder = np.zeros((self.n_vms - self.current_vms, 7))
        self.demands = np.vstack((self.demands_before, self.vm_placeholder))

        self.current_step = 0
        self.edges = np.expand_dims(np.arange(self.n_vms + self.n_pms), axis=-1)
        for vm_id, vm in enumerate(self.vms):
            pm_id = vm.pm.index
            self.edges[vm_id + self.n_pms, 0] = pm_id
        self.get_obs_()
        self.reward = 0
        self.done = False
        self.info = {}
        return self.state

    def step(self, action):
        done = False

        assert self.action_space.contains(action)
        pm_state = self.current_pm_state
        demand = self.demands[action[0]][1:5]
        real = self.pms[action[1]].can_place(self.vms[action[0]])

        if real != self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4), np.round(demand, 4)):
            import pdb
            pdb.set_trace()

        vm = self.vms[action[0]]  # 要迁移的虚机
        pm_dest = self.pms[action[1]]  # 迁移的目的地物理机

        pm_source = vm.pm  # 迁移的虚机的源物理机
        pm_source_id = vm.pm.index  # 迁移的虚机的源物理机id

        assert real == self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4),
                                           np.round(demand, 4)), \
            f"real = {real}, can_pm_meet_vm = {self.can_pm_meet_vm(np.round(pm_state[action[1], :4], 4), np.round(demand, 4))}"

        if pm_source_id == action[1]:
            reward = 0
        elif not real:
            print(f'real: {real}, is_source = {pm_source_id == action[1]}')
            pm_mask = self.get_pm_mask(int(action[0]))
            print('all_mask: ', pm_mask)
            print('pm_mask: ', pm_mask[int(action[1])])
            print('real: ', self.pms[action[1]].can_place(self.vms[action[0]]))
            raise ValueError('PM action is not fully masked. Improper action selected!')
        else:
            frag_rate_before_mig = self.get_fragment_rate_reward([pm_source, pm_dest])
            self.pm_add_vm(pm_state[action[1], :], demand, pm_state[pm_source_id, :], pm_dest, vm, pm_source,
                           self.demands)

            frag_rate_after_mig = self.get_fragment_rate_reward([pm_source, pm_dest])
            reward = (sum(frag_rate_before_mig) - sum(frag_rate_after_mig)) / 4

        self.current_step += 1

        if self.current_step >= self.MAX_STEPS:
            done = True

        self.edges[action[0] + self.n_pms] = action[1]
        self.get_obs(pm_state)

        return self.state, reward, done, {
            "fragment_rate": self.scheduler.get_fragment_rate()}  # "init_frag_rate": self.init_frag_rate

    def get_obs_(self):
        self.current_pm_state = np.hstack((self.all_pm_free_cpu / 88, self.all_pm_free_mem / 368776,
                                           self.fragment_rate_numa0, self.fragment_mode_16_numa0,
                                           self.fragment_rate_numa1, self.fragment_mode_16_numa1))

        # construct vm info
        vm_state1 = np.vstack(
            [self.vm_cpu_numa0, self.vm_cpu_numa1, self.vm_mem_numa0,
             self.vm_mem_numa1, self.vm_frag_mode16_numa0, self.vm_frag_mode16_numa1]).T
        if self.normalize:
            self.state = {'pm_info': (self.current_pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.vstack(((vm_state1 - self.vm_mean) / self.vm_std, self.vm_placeholder[:, 1:])),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }
        else:
            self.state = {'pm_info': self.current_pm_state,
                          'vm_info': np.vstack((vm_state1, self.vm_placeholder[:, 1:])),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }

    def get_obs(self, pm_state):
        if self.normalize:
            self.state = {'pm_info': (pm_state - self.pm_mean) / self.pm_std,
                          'vm_info': np.vstack([(self.demands_before[:, 1:] - self.vm_mean) / self.vm_std,
                                                self.vm_placeholder[:, 1:]]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }
        else:
            self.state = {'pm_info': pm_state,
                          'vm_info': np.vstack([self.demands_before[:, 1:], self.vm_placeholder[:, 1:]]),
                          'num_steps': np.array([[self.current_step / self.MAX_STEPS]]),
                          'num_vms': self.current_vms,
                          'edges': self.edges
                          }


class Cluster:
    def __init__(self, cluster):
        self.id = cluster["cluster_name"]
        self.pms = {}  # key=pm_id, value=pm
        self.vms = {}  # key=vm_id, value=vm
        self.racks = {}  # key=rack_id, value={pm_id}
        self._parse_pm_info(cluster["host_info"])
        self._parse_vm_info(cluster["vm_instance"])
        self.allow_vm_num = self._parse_vm_type_info(cluster["vm_type_info"])
        self.pm_isolation, self.rack_isolation = self._parse_isolation_info(cluster["other_requirement"])
        for pm in self.pms.values():
            pm.update_numa_usage()

    @staticmethod
    def _parse_vm_type_info(vm_type_info):
        allow_vm_num = defaultdict(lambda: 0)
        for record in vm_type_info:
            allow_vm_num[record["vm_type"]] = record["flavor_limit"]
        return allow_vm_num

    @staticmethod
    def _parse_isolation_info(isolation_info):
        isolated_sets_on_pm = []
        isolated_sets_on_rack = []
        for record in isolation_info["deployment_set"]:
            if record["granularity"] == "host":
                isolated_sets_on_pm.append(set(record["data"]))
            else:
                isolated_sets_on_rack.append(set(record["data"]))
        return isolated_sets_on_pm, isolated_sets_on_rack

    def _parse_pm_info(self, pms_info):
        pms = self.pms
        racks = self.racks

        for pm_info in pms_info:
            pm = PhysicalMachine(pm_info, self)
            pms[pm.id] = pm
            if pm.rack_id not in racks:
                racks[pm.rack_id] = set()
            racks[pm.rack_id].add(pm.id)

    def _parse_vm_info(self, vms_info):
        for vm_info in vms_info:
            VirtualMachine(vm_info, self.pms[vm_info["host_id"]])

    def get_involved_pms(self):
        return filter(lambda pm: pm.involved, self.pms.values())

    def query_max_allow(self, vm_type):
        return self.allow_vm_num[vm_type]

    def check_feasibility(self, all_pms):
        # 机架部署集检查
        for _, pm_ids in self.racks.items():
            vms_on_this_rack = []
            for pm_id in pm_ids:
                vms_on_this_rack.extend(list(all_pms[pm_id].vms.values()))
            for vm in vms_on_this_rack:
                for another_vm in vms_on_this_rack:
                    if another_vm is vm:
                        continue
                    assert another_vm.id not in vm.rack_excluding, f"conflict: {vm.id} and {another_vm.id}"

        # 集群实例最大数量约束
        type_num = defaultdict(lambda: 0)
        for vm in self.vms.values():
            type_num[vm.type] += 1
        for vm_type, num in type_num.items():
            assert num <= self.allow_vm_num[vm_type]

        # 物理机检查
        for pm in self.pms.values():
            pm.check_feasibility()


class Const:
    def __init__(self, *args, **kws):
        pass

    # Virtual Machine Scheduling Constants
    # 1. physical machine states
    ACTIVE = 1
    DISABLED = 2
    OUTAGE = 3

    # 2. optimizing objectives
    MORE_VM = 0
    LESS_PM = 2
    LESS_FRAGMENT = 1

    # 3. return code
    INFEASIBLE = {"status_code": 500, "error_info": "solution failed"}


class PhysicalMachine:
    states = {"ACTIVE": Const.ACTIVE, "OUTAGE": Const.OUTAGE, "DISABLED": Const.DISABLED}

    def __init__(self, pm_info, cluster):
        self.id = pm_info["node_id"]
        self.rack_id = pm_info["rack_id"]
        self.status = self.states[pm_info["status"]]
        self.cluster = cluster
        self.numas = {}
        self.vms = {}
        for numa_info in pm_info["numa"]:
            self.numas[numa_info["node"]] = NUMA(numa_info, self)
        self.cpu = sum(numa.cpu for numa in self.numas.values())
        self.mem = sum(numa.mem for numa in self.numas.values())
        self.involved = False
        self.migrate_in = []  # 记录第二阶段之后要迁入的虚拟机实例
        self.migrate_out = []  # 记录第二阶段之后要迁出的虚拟机实例

    def is_active(self):
        return self.status == Const.ACTIVE

    def add_a_vm(self, vm, deploy_numa=None):
        if deploy_numa is None:
            deploy_numa = vm.deploy_numa
        else:
            vm.deploy_numa = deploy_numa

        # print("deploy_numa is", deploy_numa)
        self.vms[vm.id] = vm
        if self.cluster is not None:
            self.cluster.vms[vm.id] = vm
        vm.pm = self
        # print(vm.numa_coeff)
        # vm.numa_coeff = 1
        # set vm.numa_coeff = 1
        for numa_id in deploy_numa:
            self.numas[numa_id].free_cpu -= vm.cpu * vm.numa_coeff
            self.numas[numa_id].free_mem -= vm.mem * vm.numa_coeff

    def release_a_vm(self, vm, deploy_numa=None):
        # import pdb
        # pdb.set_trace()
        if deploy_numa is None:
            deploy_numa = vm.deploy_numa

        self.vms.pop(vm.id)
        if self.cluster is not None:
            self.cluster.vms.pop(vm.id)
        vm.pm = None
        for numa_id in deploy_numa:
            self.numas[numa_id].free_cpu += vm.cpu * vm.numa_coeff
            self.numas[numa_id].free_mem += vm.mem * vm.numa_coeff

    def can_place(self, vm):
        # 仅在第三阶段使用
        if vm.double_numa:
            return (min(self.numas[i].free_cpu for i in range(2)) >= vm.cpu // 2 and
                    min(self.numas[i].free_mem for i in range(2)) >= vm.mem // 2)
        else:
            return ((self.numas[0].free_cpu >= vm.cpu and self.numas[0].free_mem >= vm.mem) or
                    (self.numas[1].free_cpu >= vm.cpu and self.numas[1].free_mem >= vm.mem))

    def can_place1(self, vm):
        # 已修改适配RL task
        if vm.double_numa:
            if min(self.numas[i].free_cpu for i in range(2)) >= vm.cpu // 2 and min(
                    self.numas[i].free_mem for i in range(2)) >= vm.mem // 2:
                return [0, 1]
            return []
        else:
            if self.numas[0].free_cpu >= vm.cpu and self.numas[0].free_mem >= vm.mem:
                return [0]
            if self.numas[1].free_cpu >= vm.cpu and self.numas[1].free_mem >= vm.mem:
                return [1]

    def get_total_free_cpu(self):
        return sum(numa.free_cpu for numa in self.numas.values())

    def get_total_free_mem(self):
        return sum(numa.free_mem for numa in self.numas.values())

    def get_extra_deploy(self, cpu, double):

        if double:
            cpu //= 2
            for numa in self.numas.values():
                assert numa.free_cpu >= 0, f"numa.free_cpu = {numa.free_cpu}"
            return min(numa.free_cpu // cpu for numa in self.numas.values())
        else:
            for numa in self.numas.values():
                # info = f"numa.free_cpu = {numa.free_cpu}"
                # print(info)
                assert numa.free_cpu >= 0, f"numa.free_cpu = {numa.free_cpu}"
                # import pdb
                # pdb.set_trace()
            return sum(numa.free_cpu // cpu for numa in self.numas.values())

    def update_numa_usage(self):
        for numa in self.numas.values():
            numa.calc_init_usage()

    def check_feasibility(self):
        # 资源约束检查
        numa_cpu = [0, 0]
        numa_mem = [0, 0]
        for vm in self.vms.values():
            assert vm.pm is self
            for numa_id in vm.deploy_numa:
                numa_cpu[numa_id] += vm.cpu * vm.numa_coeff
                numa_mem[numa_id] += vm.mem * vm.numa_coeff
        for numa in self.numas.values():
            assert abs(numa.cpu - numa_cpu[numa.id] - numa.free_cpu) <= 1e-6 and numa.free_cpu >= 0
            assert abs(numa.mem - numa_mem[numa.id] - numa.free_mem) <= 1e-6 and numa.free_mem >= 0

        # 物理机反亲和检查
        for vm in self.vms.values():
            for vm_id in self.vms:
                if vm_id == vm.id:
                    continue
                assert vm_id not in vm.host_excluding

    def get_free_cpu_arr(self):
        return np.array([numa.free_cpu for numa in self.numas.values()])


class NUMA:
    def __init__(self, numa_info, pm):
        self.id = numa_info["node"]
        self.pm = pm
        self.cpu = self.free_cpu = numa_info["total_cpu"]
        self.mem = self.free_mem = numa_info["memory_huge_total_mb"]
        self.init_cpu_use = self.init_mem_use = None

    def calc_init_usage(self):
        self.init_cpu_use = self.cpu - self.free_cpu
        self.init_mem_use = self.mem - self.free_mem

    def can_place(self, vm):
        return self.free_cpu >= vm.cpu and self.free_mem >= vm.mem


class VirtualMachine:
    def __init__(self, vm_info, pm, involved=False, subtype=None):
        self.id = vm_info["instance_id"]
        self.type = vm_info["instance_type"]
        self.src_pm = self.pm = pm
        self.allow_migration = vm_info["allow_migration"]
        self.deploy_numa = []  # 部署的numa id

        total_cpu = total_mem = 0
        for numa_info in vm_info["numa"]:
            self.deploy_numa.append(numa_info["node"])
            total_cpu += numa_info["cpu"]
            total_mem += numa_info["mem_mb"]

        self.cpu = total_cpu
        self.mem = total_mem
        self.double_numa = len(self.deploy_numa) == 2
        self.numa_coeff = 0.5 if self.double_numa else 1
        self.involved = involved
        self.subtype = subtype  # type + numa，指向subtype对象，第一阶段赋值
        self.conflict_num = -1  # 该虚拟机与其他虚拟机发生host冲突和rack冲突的加权数，-1表示未赋值
        self.origin_pm_id = pm.id if pm is not None else ""
        self.origin_numa = self.deploy_numa.copy()
        self.host_excluding = set()  # 与该实例不能共存于一台物理机的虚拟机id集合
        self.rack_excluding = set()  # 与该实例不能共存于一个机架的虚拟机id集合
        self.migrate_batch = -1  # -1表示不迁移

        if pm is not None:
            pm.add_a_vm(self)

    def is_migratable(self):
        return self.allow_migration and self.pm.is_active()

    def is_possibly_move(self):
        return self.subtype.out_numa[self.pm.i, self.deploy_numa[0]] > 0

    def calc_conflict_num(self, alpha):
        rack_conflict_num = len(self.rack_excluding)
        host_conflict_num = len(self.host_excluding)
        self.conflict_num = alpha * rack_conflict_num + (1 - alpha) * host_conflict_num

    def y_numa(self, pm_k, numa_j):
        return 1 if self.pm.i == pm_k and numa_j in self.deploy_numa else 0

    def y_pm(self, pm_k):
        return 1 if self.pm.i == pm_k else 0


class VirtualMachineSubtype:
    def __init__(self, i, vm_type, double_numa, cpu, mem, num_pm, c=1):
        # fixed attributes
        self.i = i
        self.type = vm_type
        self.cpu = cpu
        self.mem = mem
        self.double_numa = double_numa
        self.numa_num = 2 if double_numa else 1
        self.numa_coeff = 1 / self.numa_num
        self.c = c  # 迁移成本
        self.f = 0  # 权重，创建之后修改
        self.d_new = 0  # 需新建数量，创建之后修改

        # variable attributes
        # 一阶段
        # TODO: 适应各物理机有不同numa数量的场景（非欧空间）；适应numa的node_id非零始连续场景
        self.x_numa = np.zeros((num_pm, 2), dtype=int)  # x_numa[k, j]: 本虚机类型在物理机k的numa-j的最优分布数量
        self.y_numa = np.zeros((num_pm, 2), dtype=int)  # y_numa[k, j]: 本虚机类型在物理机k的numa-j的初始分布数量
        self.y_pm = np.zeros(num_pm)  # y_pm[k]: 本虚机类型在物理机k的初始分布数量
        self.d = 0  # 该类型虚机的已创建总数
        self.not_migratable = np.zeros((num_pm, 2))  # 本虚机类型在物理机k的numa-j的不可迁移数量

        # 二阶段
        self.in_numa = np.zeros((num_pm, 2), dtype=int)  # in_numa[k, j]: 本虚机类型从物理机k的numa-j的迁入数量
        self.out_numa = np.zeros((num_pm, 2), dtype=int)  # out_numa[k, j]: 本虚机类型从物理机k的numa-j的迁出数量
        self.possibly_move_vms = []  # 第二阶段中可能移动的虚机实例
        self.certainly_move_vms = []  # 第二阶段中按照冲突数确定移动的虚机实例

    def add_init_deploy(self, vm):
        for numa_i in vm.deploy_numa:
            self.y_numa[vm.pm.i, numa_i] += 1
        self.y_pm[vm.pm.i] += 1
        self.d += 1

        if not vm.involved:
            for j in vm.deploy_numa:
                self.not_migratable[vm.pm.i, j] += 1

    def bonding_optimal_deploy(self, x_numa):
        # 最优部署
        for k in range(self.x_numa.shape[0]):
            for j in range(self.x_numa.shape[1]):
                self.x_numa[k, j] = round(x_numa[self.i, k, j].x)
                # old version: self.x_numa[k, j] = x_numa[self.i, k, j].x
                # x_numa[self.i, k, j].x是浮点型，实际integer值为2时，该浮点数可能为1.99999999992，
                # 而self.x_numa[k, j]是整型，1.99999999992赋给该整型变量时，会退化为整型数1，导致错误

        # 计算从初始部署到最优部署的迁入/迁出数量
        self.in_numa = self.x_numa - self.y_numa
        self.in_numa = np.where(self.in_numa < 0, 0, self.in_numa)
        self.out_numa = self.y_numa - self.x_numa
        self.out_numa = np.where(self.out_numa < 0, 0, self.out_numa)

    def determine_move_vms(self):
        if not self.possibly_move_vms:
            return

        self.possibly_move_vms.sort(key=lambda vm: vm.conflict_num)
        out_numa = self.out_numa.copy()
        for vm in self.possibly_move_vms:
            if out_numa[vm.pm.i, vm.deploy_numa[0]] == 0:
                continue
            for numa_id in vm.deploy_numa:
                out_numa[vm.pm.i, numa_id] -= 1
            self.certainly_move_vms.append(vm)
            vm.pm.release_a_vm(vm)  # 第一次释放，老pm，老numa
        assert out_numa.min() == 0, "存在机器过迁"
        assert out_numa.max() == 0, "存在机器没迁"

    def new_target_vms(self):
        for i in range(self.d_new):
            self.certainly_move_vms.append(
                VirtualMachine(vm_info=self.create_json(i), pm=None, involved=True, subtype=self)
            )

    def create_json(self, index):
        json_dict = {
            "instance_id": f"target_vm_subtype{self.i}_index{index}",
            "instance_type": self.type,
            "host_id": "",
            "numa": [
                {"node": -node_id, "cpu": self.cpu // self.numa_num, "mem_mb": self.mem // self.numa_num}
                for node_id in range(1, 1 + self.numa_num)
            ],
            "allow_migration": True
        }
        return json_dict

    def move_with_least_conflict(self, vm, scheduler, rack_sets, pm_sets):
        pm_is, numa_is = np.where(self.in_numa > 0)  # 找出还可以放的位置
        best_seat = [(-1, -1), 10 ** 9, set()]  # 目标host+numa，冲突数, 发生冲突的物理机id集合
        seats = zip(set(pm_is), [-1] * len(set(pm_is))) if self.double_numa else zip(pm_is, numa_is)
        assert len(pm_is), "有虚机放不下"

        for pm_i, numa_i in seats:
            conflict_num = 0
            conflicted_pms = set()

            # 计算rack冲突
            this_rack_id = scheduler.active_pms[pm_i].rack_id
            for rack_set in rack_sets:
                if vm.id not in rack_set:
                    continue
                for co_vm_id in rack_set:
                    if co_vm_id == vm.id or scheduler.vms[co_vm_id].pm is None:
                        continue
                    that_rack_id = scheduler.vms[co_vm_id].pm.rack_id
                    if this_rack_id == that_rack_id:
                        conflict_num += 1
                        conflicted_pms |= {scheduler.active_pms[pm_i].id, scheduler.vms[co_vm_id].pm.id}
            # 计算pm冲突
            this_pm_id = scheduler.active_pms[pm_i].id
            for pm_set in pm_sets:
                if vm.id not in pm_set:
                    continue
                for co_vm_id in pm_set:
                    if co_vm_id == vm.id or scheduler.vms[co_vm_id].pm is None:
                        continue
                    that_pm_id = scheduler.vms[co_vm_id].pm.id
                    if this_pm_id == that_pm_id:
                        conflict_num += 1
                        conflicted_pms.add(this_pm_id)

            if conflict_num < best_seat[1]:
                best_seat = [(pm_i, numa_i), conflict_num, conflicted_pms]
            if conflict_num == 0:
                break

        (pm_i, numa_i), _, conflicted_pms = best_seat
        if self.double_numa:
            self.in_numa[pm_i, 0] -= 1
            self.in_numa[pm_i, 1] -= 1
            vm.deploy_numa = [0, 1]
        else:
            self.in_numa[pm_i, numa_i] -= 1
            vm.deploy_numa = [numa_i]  # 设置新numa
        scheduler.active_pms[pm_i].add_a_vm(vm)  # 第一次绑定，新pm，新numa
        # print(f"{vm.id} moves to {(pm_i, numa_i)}, conflict num = {conflict_num}")  # debug用

        return conflicted_pms


class VirtualMachineScheduler:
    def __init__(self, clusters, vm_types, requirements):
        self.clusters = {cluster["cluster_name"]: Cluster(cluster) for cluster in clusters}
        self.vm_types = {vm_type["vm_type"]: vm_type for vm_type in vm_types}
        self.requirements = requirements

        self.pms = self.get_all_pms()
        self.vms = self.get_all_vms()
        self._parse_isolation_info()  # 把反亲和、部署集信息绑定为相应虚机对象的属性
        self._merge_racks()  # 合并不同集群下的同一个机架
        self.active_pms = list(filter(lambda pm: pm.is_active(), self.pms.values()))

        for vm in self.vms.values():
            vm.allow_migration = True

        self.migratable_vms = list(filter(lambda vm: vm.is_migratable(), self.vms.values()))

        self.p1_model = None
        self.p2_model = None
        self.p3_model = None

        # import pdb
        # pdb.set_trace()

    def get_all_pms(self):
        pms = {}
        for cluster in self.clusters.values():
            pms.update(cluster.pms)
        return pms

    def get_all_vms(self):
        vms = {}
        for cluster in self.clusters.values():
            # print(len(cluster.vms))
            vms.update(cluster.vms)

        # import pdb
        # pdb.set_trace()

        return vms

    def get_all_isolation_sets(self, rack=True):
        all_sets = []
        for cluster in self.clusters.values():
            all_sets.extend(cluster.rack_isolation if rack else cluster.pm_isolation)
        return all_sets

    def _merge_racks(self):
        # 处理"一个机架分属于多个集群"的情况
        for cluster1 in self.clusters.values():
            for cluster2 in self.clusters.values():
                if cluster1 is cluster2:
                    continue
                if not cluster1.racks.keys() & cluster2.racks.keys():
                    continue
                for key in cluster1.racks:
                    if key in cluster2.racks:
                        cluster1.racks[key] |= cluster2.racks[key]
                        cluster2.racks[key] = cluster1.racks[key]

    def _parse_isolation_info(self):
        vms = self.vms

        for host_set in self.get_all_isolation_sets(rack=False):
            for vm_id in host_set:
                vms[vm_id].host_excluding |= host_set

        for rack_set in self.get_all_isolation_sets(rack=True):
            for vm_id in rack_set:
                vms[vm_id].rack_excluding |= rack_set

        for vm in self.vms.values():
            if vm.host_excluding:
                vm.host_excluding.remove(vm.id)
            if vm.rack_excluding:
                vm.rack_excluding.remove(vm.id)

    def change_optimization_range(self):
        # modify `self.active_pms` and `self.migratable_vms`
        # not implemented
        pass

    def clear_flag(self):
        for pm in self.active_pms:
            if hasattr(pm, "i"):
                delattr(pm, "i")
            pm.involved = False
        for vm in self.migratable_vms:
            vm.involved = False

    def get_objective_weights(self):
        if self.requirements["optimization_objective"] == Const.MORE_VM:
            info("当前优化目标为<部署更多虚拟机>")
            return 1, 0, 0  # wr, wf, wp
        elif self.requirements["optimization_objective"] == Const.LESS_PM:
            info("当前优化目标为<腾空更多物理机>")
            return 1, 0.01, 100
        elif self.requirements["optimization_objective"] == Const.LESS_FRAGMENT:
            info("当前优化目标为<降低碎片率>")
            return 1, 100 * len(self.vms), 0
        else:
            raise ValueError("未知优化目标")

    def get_used_pm_num(self):
        # 计算当前的物理机占用数量
        return sum([min(len(pm.vms), 1) for pm in self.active_pms])

    def get_fragment_rate(self):
        # 计算当前的碎片率
        res = 0
        free_cpu = sum(pm.get_total_free_cpu() for pm in self.active_pms)
        for vm_type in self.vm_types.values():
            cpu = vm_type["request_cpu"]
            fragment = free_cpu - cpu * sum(pm.get_extra_deploy(cpu, vm_type["numa"] == 2) for pm in self.active_pms)
            res += vm_type["weight"] * fragment / free_cpu

        return res
