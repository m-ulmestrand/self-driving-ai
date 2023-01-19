'''
Simple script for initialising which device neural network calculations are done on.

Author: Mattias Ulmestrand
'''


import torch


def init_cuda(device):
    if torch.cuda.is_available() and device[:4].lower() == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device
