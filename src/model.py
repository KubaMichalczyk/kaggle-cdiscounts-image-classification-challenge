import torch.nn as nn
from torchvision import models

def get_model(pretrained):

    if pretrained:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)

    model.fc = nn.Linear(in_features=2048, out_features=5270, bias=True)

    return model
