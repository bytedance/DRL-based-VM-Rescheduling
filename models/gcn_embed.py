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
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv


logger = logging.getLogger('VM.gcn_embed')


class GCN_Embedder(nn.Module):
    def __init__(self, params):
        super(GCN_Embedder, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        self.d_hidden = params.d_hidden  # 8
        self.num_pm = params.num_pm
        self.pm_cov = params.pm_cov  # 6
        self.output_dim = self.d_hidden  # 8

        self.conv1 = GATv2Conv(params.pm_cov + 1, params.d_hidden, num_heads=2)
        self.conv2 = GATv2Conv(params.d_hidden * 2, self.output_dim, num_heads=1, feat_drop=0.1, attn_drop=0.1)

    def forward(self, g, in_feat, b_sz):
        h = self.conv1(g, in_feat)
        h = F.elu(h).reshape(-1, self.d_hidden * 2)
        h = self.conv2(g, h).reshape(b_sz, -1, self.output_dim)
        return h[:, :self.num_pm], h[:, self.num_pm:]


class GCN_Wrapper(nn.Module):
    def __init__(self, params):
        super(GCN_Wrapper, self).__init__()
        self.model = GCN_Embedder(params).to(params.device)
