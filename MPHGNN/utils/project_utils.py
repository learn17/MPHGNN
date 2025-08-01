from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
from MPHGNN.global_configuration import global_config

def torch_normalize_l2(x):
    return F.normalize(x, dim=-1)

torch_normalize = torch_normalize_l2

def get_reversed_etype(etype):
    
    if etype[0] == etype[2]:
        return etype

    reversed_etype_ = etype[1][2:] if etype[1].startswith("r.") else "r.{}".format(etype[1])
    return (etype[2], reversed_etype_, etype[0])

def create_func_torch_random_project_create_kernel_circulant(stddev=1.0):
    def torch_random_project_create_kernel_circulant(x, units, input_units=None, generator=None):
        if input_units is None:
            input_units = x.size(-1)
        shape = [input_units, units]  


        if generator is None:
            c0 = torch.randn(input_units) * stddev
        else:
            c0 = torch.randn(input_units, generator=generator) * stddev


        kernel = torch.stack([torch.roll(c0, shifts=i) for i in range(units)], dim=1) 

        return kernel
  
    return torch_random_project_create_kernel_circulant


def torch_random_project_common(x, units, activation=False, norm=True, kernel=None, generator=None):


    if kernel is None:
        kernel = global_config.torch_random_project_create_kernel(x, units, generator=generator)

    h = x @ kernel

    if norm:
        h = torch_normalize(h)


    return h


global_config.torch_random_project = torch_random_project_common
global_config.torch_random_project_create_kernel = create_func_torch_random_project_create_kernel_circulant(stddev=1.0)


def torch_random_project_then_sum(x_list, units, norm=True, generator=None):

    h_list = [global_config.torch_random_project(x, units, norm=norm, generator=generator) 
        for x in x_list]

    h = torch.stack(h_list, dim=0).sum(dim=0)

    return h


def torch_random_project_then_mean(x_list, units, norm=True, num_samplings=None):

    h_list = [global_config.torch_random_project(x, units, norm=norm) 
        for x in x_list]
        
    h = torch.stack(h_list, dim=0).mean(dim=0)

    return h



def torch_random_project_enhanced(x, units, rounds=8, norm=True, generator=None):
    """
    Feature perturbation enhancement
    """
    B, D = x.shape
    device = x.device


    x = x.to(torch.float32)

    # Construct the circulant projection kernel
    kernel = global_config.torch_random_project_create_kernel(x, units, generator=generator).to(device).to(torch.float32)

    # 1. Original branch projection and normalization
    projected_origin = x @ kernel
    projected_origin = torch_normalize_l2(projected_origin)


    # 2. Perturbation-enhanced branch: sign flipping + column permutation
    if rounds > 0:
        with torch.amp.autocast('cuda', enabled=False):
            # Sign-flip masks
            sign_masks = (torch.randint(0, 2, (rounds, 1, D), generator=generator, device=device).float() * 2 - 1).to(torch.float32)

            perturbations = torch.empty((rounds, B, D), device=device, dtype=torch.float32)

            for i in range(rounds):
                x_flipped = x * sign_masks[i]

                perm = torch.randperm(D, generator=generator, device=device)
                x_permuted = x_flipped.index_select(dim=-1, index=perm)
                
                perturbations[i] = x_permuted

            # 3. Mean aggregation
            x_enhanced =  perturbations.mean(dim=0)

            # 4. Project the perturbation branch and normalize
            projected_enhanced = (x_enhanced @ kernel).to(dtype=x.dtype)
            projected_enhanced = torch_normalize_l2(projected_enhanced) 

    else:
        return projected_origin


    # 4. Fusion
    fused = projected_origin + projected_enhanced

    return  fused






