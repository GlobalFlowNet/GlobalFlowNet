import torch


def getGuassianKernel(span, std=None):
    x = span*torch.linspace(-1,1,2*span+1)
    if std is None:
        std = span/3    
    k = torch.exp(-x**2/(2*std**2))
    k = k/k.sum()
    return k