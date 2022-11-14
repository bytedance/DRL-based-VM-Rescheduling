## 	Deep Reinforcement Learning-based Virtual Machine Rescheduling

We are still working on this repository. A more complete and clean version will be provided soon.


### Installation Steps

1. Install Anaconda:

```
$ conda create -n rl_vm_scheduling python=3.7
$ conda activate rl_vm_scheduling
```

2. Install RLlib:

```
$ pip install gym==0.23.1
$ pip install "ray[rllib]" tensorflow torch
$ pip install -e gym-reschdule_combination
```

### Running Steps

- Train PPO-based agent
```
$ python3 main.py
```
- To use pretrained model for VM selection
```
$ python3 main.py --track --model [mlp/attn] --pretrain
```
- Evaluation
```
$ python3 eval.py --restore-name [] --restore-file-name [] --model [mlp/attn]
```

### Environments
* generalizer-v0: Base environment. Fixed number of VMs.
* generalizer-v1: Dynamic number of VMs.
* graph-v1: Dynamic number of VMs with vm-pm affiliations to support graph models.
