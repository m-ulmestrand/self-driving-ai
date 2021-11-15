import torch


def init_cuda(cuda=True):
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device
