# https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # convert PIL images to Tensors
    transform=ToTensor(),
    # convert integer labels to one-hot encoded tensors
    # one-hot encoded means each label is positionally denoted in a tensor (almost like an array mask)
    # e.g.:
    # >>> torch.zeros(10, dtype=torch.float).scatter(0, torch.tensor(0), value=1)
    # tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # >>> torch.zeros(10, dtype=torch.float).scatter(0, torch.tensor(1), value=1)
    # tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

    # >>> torch.zeros(10, dtype=torch.float).scatter(0, torch.tensor(2), value=1)
    # tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])

    # >>> torch.zeros(10, dtype=torch.float).scatter(0, torch.tensor(3), value=1)
    # tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)