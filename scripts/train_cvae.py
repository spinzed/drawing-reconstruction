import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

from vae_mode_classic import VAE
from dataset import ImageDataset
from utils import compile_img
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    #budući da se strokevi sastoje od x, y, p potrebno je strokese pretvrotiti u jednodimenzionalni vektor
    strokes, words = zip(*batch)
    print(strokes)
    strokes = [torch.tensor(stroke) for stroke in strokes]
    strokes = pad_sequence(strokes)
    print(words)
    return torch.flatten(strokes)

# Global setup

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Global variables

dataset_dir = "quickdraw"
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
image_limit = 1000

config = {
    "epochs": 20,
    "lr": 10e-4,
    "weight_decay": 10e-5,
    "grad_clip": None,
}

# Helper functions
    
def eval_loss(loader, model):
    L_total = 0
    with torch.no_grad():
        for X, y_ in loader:
            X = X.to(device)
            L_total += model.loss(X)
    return L_total / len(loader)

# Train function

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    model = model.to(device)
    Ls_train = []
    Ls_val = []
    grad_norms = []
    epochs = int(config["epochs"])

    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()

            for i, (X, y_) in enumerate(train_loader):
                optimizer.zero_grad()
                
                model.loss(X_)
                
                L.backward()
                if config["grad_clip"] != None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]).cpu())
                print(f"---> {i+1}/{len(train_loader)}, loss: {L}, grad_norm: {grad_norm}")
                optimizer.step()

            L_train = eval_loss(train_loader, loss)
            L_val = eval_loss(val_loader, loss)
            Ls_train.append(L_train.cpu())
            Ls_val.append(L_val.cpu())
            
            print(f"-> Total epoch {epoch+1}/{epochs} loss_train: {L_train}, loss_val: {L_val}")

    except KeyboardInterrupt:
        print("Early stop")
    
    plt.plot(Ls_train, label="Train Losses")
    plt.plot(Ls_val, label="Val Losses")
    plt.show()


# Main

if __name__ == "__main__":
    model = VAE(28*28, 400, 200)

    dataset = ImageDataset(dataset_dir)
    print(f"Dataset loaded")

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # TODO: fix all of this below
    print(f"Train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    
    train(model, train_loader, val_loader, config)

    loss = config["loss"]

    L_train = eval_loss(train_loader, loss)
    L_val = eval_loss(val_loader, loss)
    L_test = eval_loss(test_loader, loss)

    print(f"Final losses: train - {L_train}, val - {L_val}, test - {L_test}")

    try:
        while True:
            Xhat_ = model.sample()
            d = Xhat_.shape[-1] // 3
            strokes = torch.reshape(3, d)
            img = compile_img(strokes)
            plt.imshow(img, cmap='grey')
            plt.show()

    except KeyboardInterrupt:
        print("Done")
