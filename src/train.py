import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dataset
import engine
from model import get_model

if __name__ == '__main__':

    INPUT_PATH = os.path.join('..', 'input')
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    N_EPOCHS = 2

    category_names_df = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))
    try:
        metadata = pd.read_csv(os.path.join(INPUT_PATH, 'metadata_train.csv'))
    except FileNotFoundError:
        meta_dataset = dataset.CDiscountDataset(input_path=os.path.join(INPUT_PATH, 'train.bson'))
        meta_dataset.save_metadata(os.path.join(INPUT_PATH, 'metadata_train.csv'))
        metadata = meta_dataset.metadata

    items = metadata.index.tolist()
    targets = metadata['category_id'].values

    model = get_model(pretrained=True)
    model.to(DEVICE)

    train_items, valid_items, train_targets, valid_targets = train_test_split(
        items, targets, stratify=targets, random_state=42
    )

    train_dataset = dataset.CDiscountDataset(input_path=os.path.join(INPUT_PATH, 'train.bson'),
                                             items=train_items,
                                             metadata_file=os.path.join(INPUT_PATH, 'metadata_train.csv'),
                                             random=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    valid_dataset = dataset.CDiscountDataset(input_path=os.path.join(INPUT_PATH, 'train.bson'),
                                             items=valid_items,
                                             metadata_file=os.path.join(INPUT_PATH, 'metadata_train.csv'),
                                             random=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(N_EPOCHS):
        engine.train(train_loader, model, optimizer, device=DEVICE)
        targets, probabilities = engine.evaluate(valid_loader, model, device=DEVICE)
        predictions = np.argmax(probabilities, axis=1)
        accuracy = accuracy_score(targets, predictions)
        print(f"Epoch={epoch}, accuracy score on validation set={accuracy}")
