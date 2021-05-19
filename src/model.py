import torch.nn as nn
from torchvision import models

import config

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        x = x.reshape(*x.shape, 1, 1)
        return x

def get_model(pretrained):

    if pretrained:
        model = getattr(models, config.BASE_MODEL.lower())(pretrained=True)
        model.__name__ = config.BASE_MODEL
    else:
        model = getattr(models, config.BASE_MODEL.lower())(pretrained=False)
        model.__name__ = config.BASE_MODEL

    model.fc = nn.Sequential(Reshape(),
                             nn.Conv2d(2048, 5270, 1),
                             nn.ReLU(),
                             Flatten(),
                             nn.Linear(5270, 5270),
                             nn.ReLU(),
                             nn.Linear(5270, 5270))

    return model
