# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


import logging
import torch
import torch.nn as nn
from .components.pt_transformer import Transformer


logger = logging.getLogger('VM.attn')


class PM_candidate_model(nn.Module):
    def __init__(self, params):
        super(PM_candidate_model, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        # self.seq_length = params.num_pm
        self.d_hidden = params.d_hidden

        self.pm_encode = nn.Linear(params.pm_cov + 1, params.d_hidden)
        self.vm_encode = nn.Linear(params.vm_cov, params.d_hidden)

        self.transformer = Transformer(d_model=params.d_hidden, nhead=params.num_head,
                                       num_encoder_layers=params.transformer_blocks,
                                       num_decoder_layers=params.transformer_blocks, dim_feedforward=params.d_ff,
                                       activation='gelu', batch_first=True, dropout=params.dropout,
                                       need_attn_weights=True, device=params.device)

        self.output_layer = nn.Linear(params.d_hidden, 1)

    def forward(self, chosen_vm_embed, chosen_vm_state, num_step_states, pm_states, pm_embed, pm_attn,
                return_attns=False):
        # chosen_vm_state:  torch.Size([8, 1, 14])
        transformer_output = self.transformer(src=torch.cat([num_step_states.repeat(1, 1, self.d_hidden),
                                                             self.vm_encode(chosen_vm_state) + chosen_vm_embed], dim=1),
                                              tgt=self.pm_encode(torch.cat([pm_states,
                                                                            pm_attn[:, :, None]], dim=-1)) + pm_embed)
        score = torch.squeeze(self.output_layer(transformer_output[0]))
        if return_attns:
            return score, transformer_output[1]
        else:
            return score


class PM_Detail_Attn_Wrapper(nn.Module):
    def __init__(self, params):
        super(PM_Detail_Attn_Wrapper, self).__init__()
        self.model = PM_candidate_model(params).to(params.device)
