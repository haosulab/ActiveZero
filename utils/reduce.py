"""
Author: Isabella Liu 8/8/21
Feature: Functions for distributed training
Reference: https://github.com/alibaba/cascade-stereo/blob/master/CasStereoNet/utils/experiment.py
"""

import copy
import random
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("Invalid input type for tensor2float")


@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("Invalid input type for tensor2numpy")


@make_iterative_func
def check_all_float(vars):
    assert isinstance(vars, float)


class AverageMeterDict(object):
    def __init__(self):
        self.data = None
        self.count = 0

    def update(self, x):
        check_all_float(x)
        self.count += 1
        if self.data is None:
            self.data = copy.deepcopy(x)
        else:
            for k1, v1 in x.items():
                if isinstance(v1, float):
                    self.data[k1] += v1
                elif isinstance(v1, tuple) or isinstance(v1, list):
                    for idx, v2 in enumerate(v1):
                        self.data[k1][idx] += v2
                else:
                    assert NotImplementedError(
                        "error input type for update AvgMeterDict"
                    )

    def mean(self):
        @make_iterative_func
        def get_mean(v):
            return v / float(self.count)

        return get_mean(self.data)


def reduce_scalar_outputs(scalar_outputs, local_device):
    # local_device = scalar_outputs['loss'].device  # TODO
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            if isinstance(scalar_outputs[k], (list, tuple)):
                for sub_var in scalar_outputs[k]:
                    if not isinstance(sub_var, torch.Tensor):
                        sub_var = torch.as_tensor(sub_var, device=local_device)
                    names.append(k)
                    scalars.append(sub_var)
            else:
                if not isinstance(scalar_outputs[k], torch.Tensor):
                    scalar_outputs[k] = torch.as_tensor(
                        scalar_outputs[k], device=local_device
                    )
                names.append(k)
                scalars.append(scalar_outputs[k])

        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size

        reduced_scalars = defaultdict(list)
        for name, scalar in zip(names, scalars):
            reduced_scalars[name].append(scalar)

    return dict(reduced_scalars)
