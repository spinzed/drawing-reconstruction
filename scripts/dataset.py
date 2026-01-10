import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
import os
import utils as ut
import cv2

def erode_image(img, kernel_size=3, iterations=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = img.astype(np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


class ImageDataset(Dataset):
    def __init__(self, dirname, image_shape, image_limit, item=None):
        super().__init__()
        filter_name = "ndjson" if item is None else f"{item}.ndjson"
        filenames = [i for i in os.listdir(dirname) if i.endswith(filter_name)]

        self.dirname = dirname
        self.filenames = filenames
        self.loaded = np.hstack([ut.load_ndjson(f"{dirname}/{i}") for i in filenames])
        self.total_count = len(self.loaded)

        if len(self.loaded) > image_limit:
            self.loaded = self.loaded[:image_limit]

        self.count = len(self.loaded)
        self.image_shape = image_shape

        print(f"Found {len(filenames)} files, in total {self.count} entires")

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        entry = self.loaded[i] # https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file#the-raw-moderated-dataset
        strokes = entry["drawing"] # (num_strokes, 2 (for x and y), num_keypoints_per_stroke)
        word = entry["word"]

        percentage = np.random.random() # [0, 1>
        num_strokes = max(int(len(strokes) * percentage), 1)
        x = ut.compile_img(strokes, shape=self.image_shape, end=num_strokes)
        y = ut.compile_img(strokes, shape=self.image_shape)
        y = erode_image(y)
        return torch.tensor(x), torch.tensor(y), word
