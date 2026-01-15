import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

from models.vae_model_classic_bernoulli import VAE
from data.dataset import ImageDataset

# ---------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------

dataset_dir = "quickdraw"
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
image_limit = 10000
image_size = (256, 256)
binarization_threshold = 0.7
model_weights_save_path = "weights.pth"
#item = "cat"
item = "ice cream"
model_residual = False
latent_dim  = 200

config = {
    "epochs": 20000,
    "lr": 1e-3,
    "weight_decay": 1e-10,
    "grad_clip": 10,
    "loss": VAE.loss
    #"loss": nn.L1Loss(),
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_y(x, out):
    if model_residual:
        return torch.min(x, 1-out)
    return out

def eval_loss(loader, loss):
    model.eval()

    L_total = 0
    total_samples = 0

    with torch.no_grad():
        for X, y_, cats in loader:            
            X = y_.to(device, dtype=torch.float32)
            X = X.unsqueeze(1)
            y_ = y_.to(device, dtype=torch.float32)
            enc_mean, enc_logvar, dec_mean, dec_logvar = model(X)
            batch_size = X.size(0)
            L_total += loss(X, enc_mean, enc_logvar, dec_mean, dec_logvar).item() * batch_size
            total_samples += batch_size

    return L_total / total_samples

def visualize(axarr, images, titles):
    assert len(images) == len(titles)
    for ax in axarr:
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')

    for i, image in enumerate(images):
        axarr[i].imshow(image, cmap="grey")
        axarr[i].set_title(titles[i])

def binarize(img):
    binary = np.ones_like(img, dtype=np.float32)
    binary[img < binarization_threshold] = 0
    return binary

def save_model(model):
    torch.save(model.state_dict(), model_weights_save_path)
    print(f"Model weights saved to {model_weights_save_path}")

# ---------------------------------------------------------------------
# Train function
# ---------------------------------------------------------------------

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    model = model.to(device)
    Ls_train = []
    Ls_val = []
    grad_norms = []
    epochs = int(config["epochs"])

    plt.ion()
    n_graphs = 4
    f, axarr = plt.subplots(1, n_graphs, figsize=(3 * n_graphs, 4))
    t = time.time()
    loss = config["loss"]

    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()

            for i, (X_imgs, y_, cats) in enumerate(train_loader):
                optimizer.zero_grad()                
                
                X = y_.to(device, dtype=torch.float32)
                X = X.unsqueeze(1)                

                enc_mean, enc_logvar, dec_mean, dec_logvar = model(X)
                L = loss(X, enc_mean, enc_logvar, dec_mean, dec_logvar)
                
                L.backward()
                if config["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                grad_norm = np.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()]).cpu())
                optimizer.step()

                #print(f"---> {i+1}/{len(train_loader)}, loss: {L:.6f}, grad_norm: {grad_norm:.6f}")

            L_train = eval_loss(train_loader, loss)
            L_val = eval_loss(val_loader, loss)
            Ls_train.append(L_train)
            Ls_val.append(L_val)

            print(f"-> Total epoch {epoch+1}/{epochs} loss_train: {L_train:.6f}, loss_val: {L_val:.6f}")

            img_original = y_[0].cpu().detach().numpy()
            img_new = model.sample()
            img_new = img_new.squeeze(0).squeeze(0)
            img_new = img_new.detach().cpu().numpy()
            
            if time.time() - t >= 1:
                t = time.time()
                try:
                    f.canvas.manager.set_window_title(f"Epoch {epoch}")
                except:
                    pass
                binarized = binarize(img_new)
                visualize(axarr, [img_original, img_new, binarized], ["Original", "Sample", "Binarized"])
                plt.pause(0.01)
                


    except KeyboardInterrupt:
        print("Early stop")

    plt.ioff()
    plt.close()

    plt.plot(Ls_train, label="Train Losses")
    plt.plot(Ls_val, label="Val Losses")
    vals = sorted(Ls_train + Ls_val)
    cutoff = vals[int(0.5 * len(vals))] * 2
    plt.ylim(0, cutoff)
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print(image_size[0])
    model = VAE(image_size[0], latent_dim, torch.device(device))

    dataset = ImageDataset(dataset_dir, image_size, image_limit, item=item)
    print("Dataset loaded")

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    print(f"Train: {len(train_set):.6f}, val: {len(val_set):.6f}, test: {len(test_set):.6f}")

    train(model, train_loader, val_loader, config)

    loss = config["loss"]

    L_train = eval_loss(train_loader, loss)
    L_val = eval_loss(val_loader, loss)
    L_test = eval_loss(test_loader, loss)

    print(f"Final losses: train - {L_train:.6f}, val - {L_val:.6f}, test - {L_test:.6f}")

    save_model(model)

    try:
        while True:
            i = np.random.randint(0, len(test_set))
            img_partial, img_full, cat = test_set[i]
            x = img_partial.reshape((1, image_size[0] * image_size[1]))
            xd = x.to(device, dtype=torch.float32)
            y_d = img_full.to(device, dtype=torch.float32)
            with torch.no_grad():
                outd, _, _ = model(xd)
                yd = get_y(xd, outd)
                yd = yd.reshape(image_size)
            y = yd.cpu()

            L = loss(yd, y_d)
            print(f"Loss: {L}")

            img_reconstructed = y
            img_reconstructed_binary = binarize(img_reconstructed)

            n_graphs = 4
            f, axarr = plt.subplots(1, n_graphs, figsize=(3 * n_graphs, 4))
            visualize(
                axarr,
                [img_partial, img_full, img_reconstructed, img_reconstructed_binary],
                ["Partial", "Full", "Reconstructed", "Binarized"]
            )
            plt.show()

    except KeyboardInterrupt:
        print("Done")
