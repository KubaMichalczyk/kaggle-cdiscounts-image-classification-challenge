# This scripts will contain Dataset class implementation

class ClassificationDataset:
    
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self):
        pass