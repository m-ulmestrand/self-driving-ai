'''
Simple script for initialising which device neural network calculations are done on.

Author: Mattias Ulmestrand
'''


import torch


def init_cuda(cuda=True):
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device
