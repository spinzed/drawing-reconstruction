import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from evaluation_train import generate_and_save_images

from models.cvae_model import CVAE as VAE
from data.dataset import ImageDataset
import torch.optim.lr_scheduler as lr_scheduler

# ---------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------

dataset_dir = "quickdraw"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
image_limit =  25000
image_size = (64, 64)
binarization_threshold = 0.5
model_weights_save_path = "weights/mug.pth"
#item = "cat"
item = "mug"
model_residual = False
latent_dim  = 512
checkpointing = True
gamma = 0.5
step_size = 5

config = {
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 1e-9,
    "grad_clip": 0.01,
    "loss": VAE.loss
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
    L_recon_total = 0
    L_kl_total = 0
    total_samples = 0

    with torch.no_grad():
        for X, y_, cats in loader:            
            X = X.to(device, dtype=torch.float32)
            X = X.unsqueeze(1)
            y_ = y_.unsqueeze(1)
            y_ = y_.to(device, dtype=torch.float32)
            z, mu_p, logvar_p, mu_q, logvar_q, x_logits, x_prob = model(X, y_)
            batch_size = X.size(0)
            Lk, Lk_recon, Lk_kl= loss(X, y_, mu_p, logvar_p, mu_q, logvar_q, x_logits, beta=1.0) 
            Lk_kl = Lk_kl.item() * batch_size
            Lk = Lk.item() * batch_size
            Lk_recon = Lk_recon.item() * batch_size
            L_total += Lk
            L_recon_total += Lk_recon
            L_kl_total += Lk_kl
            total_samples += batch_size

    return L_total / total_samples, L_recon_total / total_samples, L_kl_total / total_samples

def visualize(axarr, images, titles):
    assert len(images) == len(titles)
    for ax in axarr:
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')

    for i, image in enumerate(images):
        im = axarr[i].imshow(image, cmap="Greys_r")
        axarr[i].set_title(titles[i])

def binarize(img, binarization_threshold=binarization_threshold):
    binary = np.ones_like(img, dtype=np.float32)
    binary[img < binarization_threshold] = 0
    return binary

def save_model(model, epoch=None, train_images_num=None):
    checkpoint = {
        "model_type": "CVAE",
        "image_size": image_size,
        "train_images": train_images_num,
        "epoch": epoch,
        "supported_classes": [item],
        "weights": model.state_dict(),
        "latent_dim": latent_dim,
    }
    path = model_weights_save_path
    if train_images_num is not None:
        path += f"_n{train_images_num}"
    if epoch is not None:
        path += f"_epoch{epoch}"
    path += ".pth"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model weights saved to {path}")

# ---------------------------------------------------------------------
# Train function
# ---------------------------------------------------------------------

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
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
    betas = np.linspace(0, 1, 10)
    #betas = np.linspace(0, 1, 5)
    #beta = 0

    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()

            if epoch < len(betas):
                beta = betas[epoch]
            else:
                beta = 1

            for i, (X_imgs, y_, cats) in enumerate(train_loader):
                optimizer.zero_grad()
                X = X_imgs.to(device, dtype=torch.float32)
                y_ = y_.to(device, dtype=torch.float32)
                X = X.unsqueeze(1)
                y_ = y_.unsqueeze(1)

                z, mu_p, logvar_p, mu_q, logvar_q, x_logits, x_prob = model(X, y_)
                
                L , _, _= loss(X, y_, mu_p, logvar_p, mu_q, logvar_q, x_logits, beta=beta)
                
                L.backward()
                if config["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in model.parameters()])).cpu()
                optimizer.step()

                #print(f"---> {i+1}/{len(train_loader)}, loss: {L:.6f}, grad_norm: {grad_norm:.6f}")

            L_train, recon_train, kl_train = eval_loss(train_loader, loss)
            L_val, recon_val, kl_val = eval_loss(val_loader, loss)
            Ls_train.append([L_train, recon_train, kl_train])
            Ls_val.append([L_val, recon_val, kl_val])
            

            scheduler.step()

            print(f"-> Total epoch {epoch+1}/{epochs} loss_train: {L_train:.6f}, loss_val: {L_val:.6f}")
            img_train_partial = y_[0]
            img_train_partial = img_train_partial.squeeze(0)
            img_train_partial = img_train_partial.cpu().detach().numpy()
            print(f"img train partial: {img_train_partial.shape}")

            img_train_reconstructed = model.sample(y_[0].unsqueeze(0))
            img_train_reconstructed = img_train_reconstructed.squeeze(0).squeeze(0)
            img_train_reconstructed = img_train_reconstructed.detach().cpu().numpy()
            img_train_reconstructed_b = binarize(img_train_reconstructed)

            X_val_imgs, y_, _ = next(iter(val_loader))
            y_val = y_[0]
            y_val = y_val.to(X.device)
            y_val = y_val.to(X.dtype)


            img_val_reconstructed = model.sample(y_val.unsqueeze(0).unsqueeze(0))
            img_val_partial = y_val.squeeze(0)
            img_val_reconstructed = img_val_reconstructed.squeeze(0).squeeze(0)
            img_val_reconstructed = img_val_reconstructed.detach().cpu().numpy()
            img_val_partial = img_val_partial.detach().cpu().numpy()
            img_val_binarized = binarize(img_val_reconstructed)

            if time.time() - t >= 1:
                t = time.time()
                try:
                    f.canvas.manager.set_window_title(f"Epoch {epoch}")
                except:
                    pass
                visualize(
                    axarr,
                    [img_train_reconstructed, img_val_partial, img_val_reconstructed, img_val_binarized],
                    ["Training Reconstructed", "Validation Partial", "Validation Reconstruction", "Binarized Reconstruction"]
                )
                plt.pause(0.01)
                print(f"Max pixel: {np.max(img_val_reconstructed)}")
                print(f"Min pixel: {np.min(img_val_reconstructed)}")

            if epoch % 5 == 0 and checkpointing:
                save_model(model, epoch=epoch, train_images_num=len(train_loader.dataset))
        
    except KeyboardInterrupt:
        print("Early stop")
    Ls_train = np.array(Ls_train)
    Ls_val = np.array(Ls_val)
    plt.ioff()
    plt.close()
    print(Ls_train[:, 0])
    print(Ls_train[:, 0])
    print(Ls_val[0])
    print()

    plt.plot(Ls_train, label="Train Losses")
    plt.plot(Ls_val[:, 0], label="Val Losses")
    vals = sorted(Ls_train[0] + Ls_val[0])
    cutoff = vals[int(0.5 * len(vals))] * 2
    plt.ylim(0, cutoff)
    plt.show()
    Ls_val = np.array(Ls_val)
    Ls_train = np.array(Ls_train)
    np.save(f"{item}_val_loss.npy", Ls_val)
    np.save(f"{item}_train_loss.npy", Ls_train)
    image_dir = f"images_{item}"
    os.makedirs(os.path.dirname(image_dir), exist_ok=True)

    generate_and_save_images(model, test_loader, image_dir, 'cuda', item, 20)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
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

    L_train, recon_train, kl_train = eval_loss(train_loader, loss)
    L_val, recon_val, kl_val = eval_loss(val_loader, loss)
    L_test, recon_test, kl_test = eval_loss(test_loader, loss)

    print(f"Final losses: train - {L_train:.6f}, val - {L_val:.6f}, test - {L_test:.6f}")

    save_model(model)
