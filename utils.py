import numpy as np
import torch
import torch.nn as nn

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        # accept floats or tensors
        if torch.is_tensor(val):
            v = val.detach()
        else:
            v = torch.tensor(val, dtype=torch.float32)

        self.val = v
        self.sum += v.item() * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(v)

    def show(self):
        # get last N values
        idx = max(len(self.losses) - self.num, 0)
        window = self.losses[idx:]

        # stack as tensors and return python float
        return torch.mean(torch.stack(window)).item()
    
    
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
            