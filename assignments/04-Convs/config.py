from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    batch_size = 256
    num_epochs = 2

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.AdamW(model.parameters(), lr=0.01)

    transforms = Compose([ToTensor()])
