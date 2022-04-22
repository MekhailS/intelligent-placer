import copy
import json
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import cv2


class BrickDataset(Dataset):
    ALL_LABELS = list(range(10))

    def __init__(self, path_csv, path_data, transform):
        self.df = pd.read_csv(path_csv)
        self.transform = copy.deepcopy(transform)
        self.path_data = path_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        capture = self.df.iloc[idx]
        filename = os.path.join(self.path_data, capture['crop_path'])
        label = int(capture['numerated_id'])

        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def image_to_numpy(image):
        return image.permute(1, 2, 0).numpy()