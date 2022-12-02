from gurobipy import Model, GRB, quicksum
import numpy as np
from gym_reschdule_combination.envs.vm_rescheduler_env import parse_input
import time


def mip_move(instance_json_file, max_migration_num=10):
    scheduler = parse_input(instance_json_file)

    pms = list(scheduler.get_all_pms().values())
    vms = list(scheduler.get_all_vms().values())
    print(f"该集群有物理机{len(pms)}台，虚拟机{len(vms)}台")
    print(f"碎片治理前，集群碎片率为{scheduler.get_fragment_rate() * 100:.2f}%")

    vms_1numa = [vm for vm in vms if not vm.double_numa]
    vms_2numa = [vm for vm in vms if vm.double_numa]

    for i, vm in enumerate(vms):
        vm.lid = i
    for i, pm in enumerate(pms):
        pm.lid = i

    num_numa = 2
    init_mat2 = np.zeros((len(vms), len(pms)))
    init_mat1 = np.zeros((len(vms), len(pms), num_numa))
    for vm in vms:
        if vm.double_numa:
            init_mat2[vm.lid, vm.pm.lid] = 1
        else:
            init_mat1[vm.lid, vm.pm.lid, vm.deploy_numa[0]] = 1


    start_tick = time.time()
    m = Model()
    x1 = m.addVars([(vm.lid, j, k) for vm in vms_1numa for j in range(len(pms)) for k in range(num_numa)], vtype=GRB.BINARY)
    x2 = m.addVars([(vm.lid, j) for vm in vms_2numa for j in range(len(pms))], vtype=GRB.BINARY)
    y = m.addVars(len(pms), num_numa, vtype=GRB.INTEGER)  # 每台物理机每个numa的剩余可部署的16core虚拟机数量
    z = m.addVars(len(pms), vtype=GRB.BINARY)

    for j, pm in enumerate(pms):
        for k in range(num_numa):
            m.addLConstr(
                quicksum([vm.cpu * x1[vm.lid, j, k] for vm in vms_1numa]) +
                quicksum([vm.cpu * x2[vm.lid, j] / num_numa for vm in vms_2numa]) + 16 * y[j, k] <=
                pm.cpu / num_numa * z[j]
            )
            m.addLConstr(
                quicksum([vm.mem * x1[vm.lid, j, k] for vm in vms_1numa]) +
                quicksum([vm.mem * x2[vm.lid, j] / num_numa for vm in vms_2numa]) <= pm.mem / num_numa * z[j]
            )
    for vm in vms:
        if vm.double_numa:
            m.addLConstr(
                quicksum(x2.select(vm.lid, "*")) == 1
            )
        else:
            m.addLConstr(
                quicksum(x1.select(vm.lid, "*")) == 1
            )

    m.addLConstr(
        quicksum([1 - x1[vm.lid, vm.pm.lid, vm.deploy_numa[0]] for vm in vms_1numa]) +
        quicksum([1 - x2[vm.lid, vm.pm.lid] for vm in vms_2numa]) <= max_migration_num
    )

    # 1. 腾空主机目标
    # m.setObjective(quicksum(z))
    # 2. 碎片治理目标（只算16core的碎片率）
    total_free = sum([pm.get_free_cpu_arr().sum() for pm in pms])
    m.setObjective(
        (total_free - quicksum(y) * 16) / total_free
    )
    end_tick = time.time()
    print(f"建模用时={end_tick - start_tick:.2f}s")

    m.optimize()
    print(f"建模用时={end_tick - start_tick:.2f}s, 求解用时={m.RunTime:.2f}s")

    assert m.status in [GRB.OPTIMAL]

    migration = 0
    for i, vm in enumerate(vms):
        for j, pm in enumerate(pms):
            if vm.double_numa and x2[vm.lid, pm.lid].x > 0.5 and vm.pm is not pm:
                vm.pm.release_a_vm(vm)
                pm.add_a_vm(vm)
                migration += 1
            if not vm.double_numa:
                for k in range(num_numa):
                    if x1[vm.lid, pm.lid, k].x > 0.5 and (vm.pm is not pm or vm.deploy_numa[0] != k):
                        vm.pm.release_a_vm(vm)
                        vm.deploy_numa = [k]
                        pm.add_a_vm(vm)
                        migration += 1
    print(f"总迁移次数={migration}")

    print(f"碎片治理后，集群碎片率为{scheduler.get_fragment_rate() * 100:.2f}%")
    return


if __name__ == '__main__':
    # json_file = "../data/reset_vm_pm_id_clear_big2.json"
    json_file = "../data/reset_vm_pm_id_clear2.json"
    mip_move(json_file, max_migration_num=100)
