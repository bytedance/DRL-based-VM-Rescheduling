# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


import logging
import os
import torch
import torch.nn as nn


logger = logging.getLogger('VM.mlp')


class VM_candidate_model(nn.Module):
    def __init__(self, params):
        super(VM_candidate_model, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        self.input_size = params.num_pm * params.pm_cov + params.vm_cov*params.num_vm  # 2246
        self.output_size = params.num_vm

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.vm_head = nn.Linear(128, self.output_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, vm_states, pm_states):

        b_sz = vm_states.shape[0]
        x = torch.cat([pm_states.reshape(b_sz, -1), vm_states.reshape(b_sz, -1)], dim=-1)
        hidden = self.layers(x)
        return self.vm_head(hidden), self.critic(hidden)


class VM_MLP_Wrapper(nn.Module):
    def __init__(self, params, pretrain=False):
        super(VM_MLP_Wrapper, self).__init__()
        self.model = VM_candidate_model(params).to(params.device)
        if pretrain:
            model_save_path1 = './saved_model_weights/model_network.ckpt'
            model_save_path2 = './saved_model_weights/model_vm_head.ckpt'
            assert os.path.isfile(model_save_path1)
            assert os.path.isfile(model_save_path2)
            self.model.layers.load_state_dict(torch.load(model_save_path1))
            self.model.vm_head.load_state_dict(torch.load(model_save_path2))
