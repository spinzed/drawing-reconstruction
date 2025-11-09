import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

from model import VAE
from dataset import ImageDataset

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
    "loss": nn.MSELoss(),
}

# Helper functions
    
def eval_loss(loader, loss):
    L_total = 0
    with torch.no_grad():
        for X, y_ in loader:
            X = X.to(device)
            y_ = y_.to(device)
            y = model(X)
            L_total += loss(y, y_)
    return L_total / len(loader)

# Train function

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config):
    loss = config["loss"]
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

                X = X.to(device)
                y_ = y_.to(device)

                y = model(X)
                L = loss(y, y_)
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
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

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
            i = np.random.randint(0, len(test_set))
            x, y_ = test_set[i]
            xd = x.unsqueeze(0).to(device)
            y_d = y_.to(device)
            with torch.no_grad():
                yd = model(xd).squeeze(0)
            y = yd.cpu()

            L = loss(yd, y_d)
            print(f"Loss: {L}")

            img_low = x[:3].permute(1,2,0)
            img_high = y_.permute(1,2,0)
            img_denoised = y.permute(1,2,0)

            f, axarr = plt.subplots(1,3, figsize=(12,4)) 
            for ax in axarr:
                ax.set_aspect('equal')
                ax.axis('off') 
            axarr[0].imshow(img_low)
            axarr[1].imshow(img_high)
            axarr[2].imshow(img_denoised)
            plt.show()

    except KeyboardInterrupt:
        print("Done")
