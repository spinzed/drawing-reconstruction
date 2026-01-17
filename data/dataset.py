import torch
from torch.utils.data import Dataset
import numpy as np
import os
import data.utils as ut
import cv2

with_strokes = True

def erode_image(img, kernel_size=3, iterations=1):
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

        if with_strokes:
            num_strokes = max(int(len(strokes) * percentage), 1)
            y = ut.compile_img(strokes, shape=self.image_shape, end=num_strokes)
            x = ut.compile_img(strokes, shape=self.image_shape)
        else:
            sequence = ut.strokes_to_relative_sequence(strokes)
            num_strokes = max(int(len(sequence) * percentage), 1)
            y = ut.compile_img_from_sequence(sequence[:num_strokes], img_shape=self.image_shape)
            x = ut.compile_img_from_sequence(sequence, img_shape=self.image_shape)
        y = erode_image(y)
        x = erode_image(x)
        return torch.tensor(x), torch.tensor(y), word

class NumpyDataset(Dataset):
    """Dataset wrapper for numpy arrays of sketch sequences.

    Expected input is an array-like where each element is a sequence of shape
    (T, >=3) with columns: dx, dy, pen_flag (or similar). A typical input is
    np.load("file.npz", encoding='latin1', allow_pickle=True)["train"].

    The dataset filters out too short/long sequences, clamps extreme values,
    converts to float32 and normalizes dx/dy by the global standard deviation.
    __getitem__ returns a tuple: (torch.tensor(sequence), length).
    """

    def __init__(self, arr, max_seq_length=200):
        super().__init__()
        # ensure we can iterate over arr even if it's an ndarray of objects
        sequences = [np.array(s, dtype=np.float32) for s in list(arr)]
        # filter and clamp
        filtered = []
        for seq in sequences:
            if seq.ndim != 2 or seq.shape[1] < 2:
                # not a valid sequence, skip
                continue
            if seq.shape[0] <= max_seq_length and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                filtered.append(seq.astype(np.float32))
        if len(filtered) == 0:
            raise ValueError("NumpyDataset: no valid sequences found after filtering")
        # normalize dx,dy by global std (append all dx/dy values)
        vals = []
        for s in filtered:
            vals.append(s[:, 0])
            vals.append(s[:, 1])
        vals = np.concatenate(vals)
        scale = np.std(vals)
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        for i in range(len(filtered)):
            s = filtered[i].copy()
            s[:, 0:2] /= scale
            filtered[i] = s
        self.data = filtered
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.from_numpy(seq), seq.shape[0]
