import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F
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

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device={DEVICE}")

    N_EPOCHS = 5
    BATCH_SIZE = 32
    LR = 5e-4

    category_names_df = pd.read_csv(os.path.join(config.INPUT_PATH, 'category_names.csv'))
    try:
        metadata = pd.read_csv(os.path.join(config.INPUT_PATH, 'metadata_train.csv'))
    except FileNotFoundError:
        meta_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'))
        meta_dataset.save_metadata(os.path.join(config.INPUT_PATH, 'metadata_train.csv'))
        metadata = meta_dataset.metadata

    items = metadata.index.tolist()
    targets = metadata['category_id'].values

    model = get_model(pretrained=True)
    model.to(DEVICE)

    train_items, valid_items, train_targets, valid_targets = train_test_split(
        items, targets, stratify=targets, random_state=42
    )

    train_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'),
                                             items=train_items,
                                             metadata_file=os.path.join(config.INPUT_PATH, 'metadata_train.csv'),
                                             random=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    valid_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'),
                                             items=valid_items,
                                             metadata_file=os.path.join(config.INPUT_PATH, 'metadata_train.csv'),
                                             random=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    comment = f"_model={model.__name__}_batch_size={BATCH_SIZE}_lr={LR}"
    tb = SummaryWriter(log_dir=os.path.join(config.LOG_PATH, current_time + comment))
    sample_images = torch.rand((BATCH_SIZE, *train_dataset[0][0].shape), device=DEVICE)
    tb.add_graph(model, sample_images)
    for epoch in range(N_EPOCHS):

        engine.train(train_loader, model, optimizer, device=DEVICE)
        targets, probabilities = engine.evaluate(train_loader, model, device=DEVICE)
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = accuracy_score(targets, predictions)
        print(f"Epoch={epoch}, accuracy score on train set={accuracy}")
        tb.add_scalar("Training accuracy", accuracy, epoch)
        loss = F.cross_entropy(probabilities, targets)
        tb.add_scalar("Training loss", loss, epoch)

        targets, probabilities = engine.evaluate(valid_loader, model, device=DEVICE)
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = accuracy_score(targets, predictions)
        print(f"Epoch={epoch}, accuracy score on validation set={accuracy}")
        tb.add_scalar("Validation accuracy", accuracy, epoch)
        loss = F.cross_entropy(probabilities, targets)
        tb.add_scalar("Validation loss", loss, epoch)
    tb.close()
