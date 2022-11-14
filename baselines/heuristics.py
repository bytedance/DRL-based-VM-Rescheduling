# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


from gym_reschdule_combination.envs.vm_rescheduler_env import parse_input
import numpy as np


def get_vm_dec_contrib(vm, cpu=16):
    # vm从原物理机，释放，减少的碎片
    free_cpu = np.array([vm.pm.numas[0].free_cpu, vm.pm.numas[1].free_cpu])
    vm_cpu = np.zeros(2)
    for i in vm.deploy_numa:
        vm_cpu[i] += vm.cpu * vm.numa_coeff
    return sum(free_cpu % cpu) - sum((free_cpu + vm_cpu) % cpu)


def get_vm_inc_contrib(pm, numa, vm, cpu=16):
    # vm绑定到pm的numa，减少的碎片
    free_cpu = np.array([pm.numas[0].free_cpu, pm.numas[1].free_cpu])
    vm_cpu = np.zeros(2)
    for i in numa:
        vm_cpu[i] += vm.cpu * vm.numa_coeff
    if (free_cpu < vm_cpu).any():
        return -1000
    return sum(free_cpu % cpu) - sum((free_cpu - vm_cpu) % cpu)


def filter(vms):
    best_choice = [-1000, None]
    for vm in vms:
        if vm.cpu > 16:
            continue
        contrib = get_vm_dec_contrib(vm)
        if contrib > best_choice[0]:
            best_choice = [contrib, vm]
    if best_choice[1] is not None:
        return best_choice[1]
    return


def scorer(pms, vm):
    best_choice = [-1000, None, None]
    for pm in pms:
        if pm is vm.pm:
            continue
        for numa in [[0], [1]] if not vm.double_numa else [[0, 1]]:
            contrib = get_vm_inc_contrib(pm, numa, vm)
            if contrib > best_choice[0]:
                best_choice = [contrib, pm, numa]
    return best_choice[1:]


def heuristic_move(instance_json_file, max_migration_num=10):
    scheduler = parse_input(instance_json_file)

    pms = list(scheduler.get_all_pms().values())
    vms = list(scheduler.get_all_vms().values())
    print(f"该集群有物理机{len(pms)}台，虚拟机{len(vms)}台")
    print(f"碎片治理前，集群碎片率为{scheduler.get_fragment_rate() * 100:.2f}%")

    for step in range(max_migration_num):
        move_vm = filter(vms)
        if move_vm is None:
            print("early stop")
            break
        frag_dec1 = get_vm_dec_contrib(move_vm)
        src_pm = move_vm.pm
        # print(src_pm.get_free_cpu_arr())
        src_pm.release_a_vm(move_vm)
        # print(src_pm.get_free_cpu_arr())

        target_pm, target_numa = scorer(pms, move_vm)
        frag_dec2 = get_vm_inc_contrib(target_pm, target_numa, move_vm)
        if frag_dec1 + frag_dec2 <= 0:
            src_pm.add_a_vm(move_vm)
            print("early stop")
            break
        target_pm.add_a_vm(move_vm, target_numa)
        print(f"第{step + 1}步，迁出减少碎片{frag_dec1}个，迁入减少碎片{frag_dec2}个")

    print(f"碎片治理后，集群碎片率为{scheduler.get_fragment_rate() * 100:.2f}%")
    return


if __name__ == '__main__':
    json_file = "../data/reset_vm_pm_id_clear_big2.json"
    heuristic_move(json_file, max_migration_num=100)
