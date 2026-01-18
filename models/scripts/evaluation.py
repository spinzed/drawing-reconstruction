import os
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from models.cvae_model import CVAE
from data.dataset import ImageDataset
from models.scripts.train_cvae import binarize

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

dataset_dir = "quickdraw"
image_size = (64, 64)
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 512
item = "mug"
binarization_threshold = 0.5

model_weights_path = (
    "/home/steffy/Desktop/Data/studenti/antonio_skara/"
    "drawing-reconstruction/weights/mug.pth_n17500_epoch30.pth"

)

image_limit = 100

# Output directory for qualitative results
output_dir = "images/mug"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

model = CVAE(image_size[0], latent_dim, device)
weights = torch.load(model_weights_path, weights_only=True)["weights"]
model.load_state_dict(weights)
model.to(device)
model.eval()

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------

dataset = ImageDataset(
    dataset_dir,
    image_size,
    image_limit=image_limit,
    item=item
)

_, _, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False
)

# ---------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------

recon_losses = []
kl_losses = []

with torch.no_grad():
    for X_imgs, y_, _ in test_loader:
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

# ---------------------------------------------------------------------
# Report metrics
# ---------------------------------------------------------------------

recon_loss = torch.cat(recon_losses).mean().item()
kl_loss = torch.cat(kl_losses).mean().item()
elbo = recon_loss + kl_loss

print("========== Test Set Evaluation ==========")
print(f"Reconstruction Loss (BCE): {recon_loss:.6f}")
print(f"KL Divergence:             {kl_loss:.6f}")
print(f"ELBO:                      {elbo:.6f}")

# ---------------------------------------------------------------------
# Qualitative evaluation (SAVE IMAGES)
# ---------------------------------------------------------------------

def visualize_results(model, loader, n=10, out_dir=output_dir):
    model.eval()

    X_imgs, y_, _ = next(iter(loader))
    x = X_imgs[:n].to(device, dtype=torch.float32).unsqueeze(1)
    y = y_[:n].to(device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mu_p, logvar_p = model.encoder_prior(x)
        z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * logvar_p)
        recon = torch.sigmoid(model.decoder(z, y))
        recon_binarized = binarize(recon.cpu().detach().numpy())

    for i in range(n):
        # Partial input
        plt.figure()
        plt.imshow(y[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"sample_{i}_{item}_partial.png"),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

        # Reconstruction
        plt.figure()
        plt.imshow(recon[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"sample_{item}_{i}_reconstruction.png"),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

        # Binarized Reconstruction
        plt.figure()
        plt.imshow(recon_binarized[i, 0], cmap="gray")
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"sample_{item}_{i}_reconstruction_binarized.png"),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

        # Ground truth
        plt.figure()
        plt.imshow(x[i, 0].cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(
            os.path.join(out_dir, f"sample_{item}_{i}_ground_truth.png"),
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

# ---------------------------------------------------------------------
# Run qualitative evaluation
# ---------------------------------------------------------------------

visualize_results(model, test_loader)
