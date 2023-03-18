from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize


class CONFIG:
    batch_size = 16
    num_epochs = 1

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

    transforms = Compose([ToTensor(), Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])])
