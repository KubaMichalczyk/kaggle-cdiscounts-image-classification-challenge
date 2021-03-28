import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


def train(data_loader, model, optimizer, device):

    model.train()

    for images, targets in tqdm(data_loader):

        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()


def evaluate(data_loader, model, device):

    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for images, targets in data_loader:

            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(images)

            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()

            final_targets.append(targets)
            final_outputs.append(outputs)

    return np.concatenate(final_targets), np.concatenate(final_outputs)
