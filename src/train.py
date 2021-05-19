import os
import copy
from datetime import datetime
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config
import dataset
import engine
from model import get_model

class RunBuilder:
    pass

class RunManager:
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_score', action='store_true')
    args = parser.parse_args()

    category_names_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'category_names.csv'))
    try:
        metadata = pd.read_csv(os.path.join(config.METADATA_PATH, 'metadata_train.csv'))
    except FileNotFoundError:
        meta_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'))
        meta_dataset.save_metadata(os.path.join(config.METADATA_PATH, 'metadata_train.csv'))
        metadata = meta_dataset.metadata

    items = metadata.index.tolist()
    targets = metadata['category_id'].values

    model = get_model(pretrained=True)
    model.to(config.DEVICE)

    augmentations = albumentations.Compose([
        albumentations.RandomCrop(160, 160, always_apply=True),
    ])

    train_items, valid_items, train_targets, valid_targets = train_test_split(
        items, targets, stratify=targets, test_size=0.2, random_state=42
    )

    if config.SAMPLE:
        train_items, _, _, _ = train_test_split(
            train_items, train_targets, stratify=train_targets, train_size=config.SAMPLE, random_state=42
        )
        valid_items, _, _, _ = train_test_split(
            valid_items, valid_targets, stratify=valid_targets, train_size=config.SAMPLE, random_state=42
        )
        print(f"Training with sample of {config.SAMPLE}")
        print("Train set no. of samples:", len(train_items))
        print("Validation set no. of samples:", len(valid_items))

    train_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'),
                                             items=train_items,
                                             metadata_file=os.path.join(config.METADATA_PATH, 'metadata_train.csv'),
                                             augmentations=augmentations,
                                             random=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)

    valid_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'),
                                             items=valid_items,
                                             metadata_file=os.path.join(config.METADATA_PATH, 'metadata_train.csv'),
                                             augmentations=augmentations,
                                             random=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    comment = f"_model={model.__name__}_batch_size={config.BATCH_SIZE}_lr={config.LR}"
    tb = SummaryWriter(log_dir=os.path.join(config.LOG_PATH, current_time + comment))
    sample_images = torch.rand((config.BATCH_SIZE, *train_dataset[0][0].shape), device=config.DEVICE)
    tb.add_graph(model, sample_images)

    best_model_wts = None
    best_accuracy = 0.0

    for epoch in range(config.N_EPOCHS):

        engine.train(train_loader, model, optimizer, lr_scheduler, device=config.DEVICE)

        if args.train_score:
            targets, probabilities = engine.evaluate(train_loader, model, device=config.DEVICE)
            predictions = torch.argmax(probabilities, dim=1)
            accuracy = accuracy_score(targets, predictions)
            print(f"Epoch={epoch}, accuracy score on train set={accuracy}")
            tb.add_scalar("Training accuracy", accuracy, epoch)
            loss = F.cross_entropy(probabilities, targets)
            tb.add_scalar("Training loss", loss, epoch)

        targets, probabilities = engine.evaluate(valid_loader, model, device=config.DEVICE)
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = accuracy_score(targets, predictions)
        print(f"Epoch={epoch}, accuracy score on validation set={accuracy}")
        tb.add_scalar("Validation accuracy", accuracy, epoch)
        loss = F.cross_entropy(probabilities, targets)
        tb.add_scalar("Validation loss", loss, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
    tb.close()

    model.load_state_dict(best_model_wts)
    torch.save(model, os.path.join(config.MODELS_PATH, current_time + comment + '.th'))
