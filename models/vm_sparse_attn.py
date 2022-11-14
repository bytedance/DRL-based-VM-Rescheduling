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
from .components.pt_transformer import TransformerSparseDecoder, TransformerSparseDecoderLayer


logger = logging.getLogger('VM.attn')


class VM_candidate_model(nn.Module):
    def __init__(self, params):
        super(VM_candidate_model, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        self.num_pm = params.num_pm
        self.num_vm = params.num_vm
        self.num_head = params.num_head

        self.d_hidden = params.d_hidden

        self.pm_encode = nn.Linear(params.pm_cov, params.d_hidden)
        self.vm_encode = nn.Linear(params.vm_cov, params.d_hidden)

        decoder_layer = TransformerSparseDecoderLayer(d_model=params.d_hidden, nhead=params.num_head,
                                                      dim_feedforward=params.d_ff, dropout=params.dropout,
                                                      activation='gelu', batch_first=True, norm_first=True,
                                                      need_attn_weights=True, device=params.device)
        self.transformer = TransformerSparseDecoder(decoder_layer=decoder_layer, num_layers=params.transformer_blocks)

        self.output_layer = nn.Linear(params.d_hidden, 1)
        self.critic_layer = nn.Linear(params.d_hidden, 1)
        self.critic_token = -torch.ones(1, 1, params.d_hidden).to(self.device)

    def forward(self, vm_states, num_step_states, pm_states, vm_pm_relation, num_vms_mask=None, return_attns=False):
        b_sz = vm_states.shape[0]
        local_mask = torch.zeros(b_sz, self.num_pm + self.num_vm + 2, self.num_pm + self.num_vm + 2,
                                 dtype=torch.bool, device=self.device)
        local_mask[:, 1:-1, 1:-1] = vm_pm_relation != vm_pm_relation[:, None, :, 0]
        tgt_key_pad_mask = torch.zeros(b_sz, 2 + self.num_pm + self.num_vm, dtype=torch.bool, device=self.device)
        tgt_key_pad_mask[:, 1 + self.num_pm:-1] = num_vms_mask
        transformer_output = self.transformer(tgt=torch.cat([num_step_states.repeat(1, 1, self.d_hidden),
                                                             self.pm_encode(pm_states), self.vm_encode(vm_states),
                                                             self.critic_token.repeat(b_sz, 1, 1).detach()], dim=1),
                                              local_mask=local_mask.repeat(self.num_head, 1, 1),
                                              tgt_key_padding_mask=tgt_key_pad_mask)
        pm_embed = transformer_output[0][:, 1:1 + self.num_pm]
        vm_embed = transformer_output[0][:, 1 + self.num_pm:-1]
        score = torch.squeeze(self.output_layer(vm_embed))
        critic_score = self.critic_layer(transformer_output[0][:, -1])
        if return_attns:
            return pm_embed, vm_embed, score, critic_score, transformer_output[1]
        else:
            return pm_embed, vm_embed, score, critic_score


class VM_Sparse_Attn_Wrapper(nn.Module):
    def __init__(self, params, pretrain=False):
        super(VM_Sparse_Attn_Wrapper, self).__init__()
        self.model = VM_candidate_model(params).to(params.device)
        if pretrain:
            model_save_path = './saved_model_weights/attn.ckpt'
            assert os.path.isfile(model_save_path)
            self.model.load_state_dict(torch.load(model_save_path))
