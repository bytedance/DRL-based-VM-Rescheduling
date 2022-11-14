# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


from gym.envs.registration import register


register(
    id="generalizer-v0",
    entry_point="gym_reschdule_combination.envs:VM_generlizer_v0",
)
register(
    id="generalizer-v1",
    entry_point="gym_reschdule_combination.envs:VM_generlizer_v1",
)
register(
    id="graph-v1",
    entry_point="gym_reschdule_combination.envs:VM_graph_v1",
)
register(
    id="graph-v2",
    entry_point="gym_reschdule_combination.envs:VM_graph_v2",
)
