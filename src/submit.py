import os
import csv
import argparse
import pandas as pd
import torch
from tqdm import tqdm

import config
import dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model_path', action='store', type=str)
    parser.add_argument('-m', '--message', action='store', type=str, default='')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    model_name = args.model_path.split("/")[-1]
    model = torch.load(args.model_path)

    test_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'test.bson'),
                                            metadata_file=os.path.join(config.INPUT_PATH, 'metadata_test.csv'),
                                            random=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # We need train_dataset to re-map categories from model targets (1-5270) back to their original ids
    train_dataset = dataset.CDiscountDataset(input_path=os.path.join(config.INPUT_PATH, 'train.bson'),
                                             items=[],
                                             metadata_file=os.path.join(config.INPUT_PATH, 'metadata_train.csv'),
                                             )

    submission_file = os.path.join(config.SUBMISSIONS_PATH, model_name + '.csv')
    if os.path.exists(submission_file):
        if args.force:
            os.remove(submission_file)
            with torch.no_grad(), open(submission_file, "a") as f:
                pd.DataFrame(columns=['_id', 'category_id']).to_csv(f, index=False, quoting=csv.QUOTE_NONNUMERIC)
                for indices, (images, _) in tqdm(test_loader):

                    images = images.to(config.DEVICE, dtype=torch.float)
                    probabilities = model(images)
                    probabilities = probabilities.detach().cpu()
                    predictions = torch.argmax(probabilities, dim=1)
                    labels = train_dataset.category_id_encoder.inverse_transform(predictions)
                    rows = list(zip(indices.numpy(), labels))
                    pd.DataFrame(rows).to_csv(f, header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
            os.system("kaggle competitions submit -c cdiscount-image-classification-challenge"
                      f" -f {submission_file} -m {args.message}")
        else:
            print(f"File {submission_file} exists. If you want to overwrite it, use -f argument.")
