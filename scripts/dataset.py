import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import cv2
import numpy as np
import os
import utils as ut

class ImageDataset(Dataset):
    def __init__(self, dirname):
        super().__init__()
        filenames = [i for i in os.listdir(dirname) if i.endswith("ndjson")]

        self.dirname = dirname
        self.filenames = filenames
        self.loaded = np.hstack([ut.load_ndjson(f"{dirname}/{i}") for i in filenames])
        self.count = len(self.loaded)

        print(f"Found {len(filenames)} files, in total {self.count} entires")

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        entry = self.loaded[i] # https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file#the-raw-moderated-dataset
        strokes = entry["drawing"]
        word = entry["word"]
        return strokes, word