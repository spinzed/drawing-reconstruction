import os
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
import numpy as np

binarization_threshold = 0.5

def binarize(img, binarization_threshold=binarization_threshold):
    binary = np.ones_like(img, dtype=np.float32)
    binary[img < binarization_threshold] = 0
    return binary

def generate_and_save_images(
    model,
    dataset_loader,
    output_dir,
    device="cuda",
    item="sample",
    n_visualize=10,
):
    """
    Evaluate CVAE on a dataset and save reconstructed images.

    Args:
        model (torch.nn.Module): Trained CVAE model
        dataset_loader (DataLoader): Test or evaluation DataLoader
        output_dir (str): Directory where images will be saved
        device (str): 'cuda' or 'cpu'
        item (str): Item name used in saved filenames
        n_visualize (int): Number of samples to visualize
    Returns:
        dict: {'recon_loss': float, 'kl_loss': float, 'elbo': float}
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    recon_losses = []
    kl_losses = []

    with torch.no_grad():
        for X_imgs, y_, _ in dataset_loader:
            x = X_imgs.to(device, dtype=torch.float32).unsqueeze(1)
            y = y_.to(device, dtype=torch.float32).unsqueeze(1)

            z, mu_p, logvar_p, mu_q, logvar_q, x_logits, _ = model(x, y)

            # Reconstruction loss
            recon = F.binary_cross_entropy_with_logits(
                x_logits, y, reduction="none"
            )
            recon = recon.flatten(1).mean(1)

            # KL divergence
            q = Normal(mu_q, torch.exp(0.5 * logvar_q))
            p = Normal(mu_p, torch.exp(0.5 * logvar_p))
            kl = kl_divergence(q, p).mean(1)

            recon_losses.append(recon.cpu())
            kl_losses.append(kl.cpu())

    # Metrics
    recon_loss = torch.cat(recon_losses).mean().item()
    kl_loss = torch.cat(kl_losses).mean().item()
    elbo = recon_loss + kl_loss

    print("========== Evaluation ==========")
    print(f"Reconstruction Loss (BCE): {recon_loss:.6f}")
    print(f"KL Divergence:             {kl_loss:.6f}")
    print(f"ELBO:                      {elbo:.6f}")

    # -----------------------------------------------------------------
    # Qualitative evaluation
    # -----------------------------------------------------------------
    X_imgs, y_, _ = next(iter(dataset_loader))
    x = X_imgs[:n_visualize].to(device, dtype=torch.float32).unsqueeze(1)
    y = y_[:n_visualize].to(device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mu_p, logvar_p = model.encoder_prior(x)
        z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * logvar_p)
        recon = torch.sigmoid(model.decoder(z, y))
        recon_binarized = binarize(recon.cpu().detach().numpy())

    for i in range(n_visualize):
        # Partial input
        plt.imshow(y[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{item}_{i}_partial.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close()

        # Reconstruction
        plt.imshow(recon[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{item}_{i}_reconstruction.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close()

        # Binarized reconstruction
        plt.imshow(recon_binarized[i, 0], cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{item}_{i}_reconstruction_binarized.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close()

        # Ground truth
        plt.imshow(x[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"{item}_{i}_ground_truth.png"),
                    bbox_inches="tight", pad_inches=0)
        plt.close()

    return {"recon_loss": recon_loss, "kl_loss": kl_loss, "elbo": elbo}
