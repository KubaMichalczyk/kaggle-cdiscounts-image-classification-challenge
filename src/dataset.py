import struct
import bson
import pandas as pd
import numpy as np
import cv2
import torch

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class CDiscountDataset:
    
    def __init__(self, input_path, items=None, metadata_file=None, image_id=0, random=False, resize=None, augmentations=None):
        
        self.input_path = input_path
        self.items = items
        self.resize = resize
        self.augmentations = augmentations
        self.metadata = None
        self.category_id_encoder = None
        self.image_id = image_id
        self.random = random
        
        if metadata_file is not None:
            self.metadata = pd.read_csv(metadata_file)
        else:
            print(f"Dataset metadata is being extracted from {self.input_path}, this may take several minutes.")
            self.extract_metadata()
        
        if items is None:
            self.items = self.metadata.index

        if self.category_id_encoder is None:
            category_ids = self.metadata['category_id'].unique()
            self.category_id_encoder = LabelEncoder()
            self.category_id_encoder.fit(category_ids)

    def extract_metadata(self, num_items=None):
        """Source: https://www.kaggle.com/vfdev5/random-item-access"""
        with open(self.input_path, 'rb') as f, tqdm(num_items) as progress_bar: 
            items = []
            offset = 0
            while True:        
                progress_bar.update()
                f.seek(offset)
                
                item_length_bytes = f.read(4) 
                if len(item_length_bytes) == 0:
                    break                
                # Decode item length:
                length = struct.unpack("<i", item_length_bytes)[0]
                
                f.seek(offset)
                item_data = f.read(length)
                assert len(item_data) == length, "%i vs %i" % (len(item_data), length)
                
                # Check if we can decode
                item = bson.BSON.decode(item_data)
                items.append({
                    'item_id': item['_id'],
                    'category_id': item['category_id'],
                    'n_images': len(item['imgs']),
                    'offset': offset,
                    'length': length,
                })
                offset += length
        progress_bar.close()
        self.metadata = pd.DataFrame(items).set_index('item_id')


    def save_metadata(self, path):
        self.metadata.to_csv(path, index=False)


    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.items)
    

    def __getitem__(self, item_id):
        assert item_id in self.metadata.index 
        assert self.image_id <= self.metadata.loc[item_id, 'n_images']
        with open(self.input_path, 'rb') as f:
            offset, length = self.metadata.loc[item_id, ['offset', 'length']]
            f.seek(offset)
            item_bytes = f.read(length)
        item_decoded = bson.BSON.decode(item_bytes)
        
        if self.random:
            selected = np.random.choice(self.metadata.loc[item_id, 'n_images'])
        else:
            selected = self.image_id
        
        image_bytes = item_decoded['imgs'][selected]['picture']
        
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), 
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        if self.resize is not None:
            image = cv2.resize(image, dsize=self.resize)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        # Torch expects channels first
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        target = self.category_id_encoder.transform([item_decoded['category_id']])[0]

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(target, dtype=torch.long),
        )


