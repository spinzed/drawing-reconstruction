import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from models.cvae_model import CVAE
from data.dataset import ImageDataset

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

dataset_dir = "quickdraw"
image_size = (64, 64)
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 512
item = "smiley face"
model_weights_path = "/home/steffy/Desktop/Data/studenti/antonio_skara/drawing-reconstruction/checkpoint_epoch_25_smiley_face.pth"

# ---------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------

model = CVAE(image_size[0], latent_dim, device)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()
image_limit = 100

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------

dataset = ImageDataset(dataset_dir, image_size, image_limit=image_limit, item=item)
_, _, test_set = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False
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
# Qualitative evaluation
# ---------------------------------------------------------------------

def visualize_results(model, loader, n=4):
    model.eval()
    X_imgs, y_, _ = next(iter(loader))
    x = X_imgs[:n].to(device, dtype=torch.float32).unsqueeze(1)
    y = y_[:n].to(device, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        mu_p, logvar_p = model.encoder_prior(x)
        z = mu_p + torch.randn_like(mu_p) * torch.exp(0.5 * logvar_p)
        recon = torch.sigmoid(model.decoder(z, y))

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for i in range(n):
        axes[i, 0].imshow(x[i, 0].cpu(), cmap="gray")
        axes[i, 0].set_title("Ground Truth")
        axes[i, 1].imshow(recon[i, 0].cpu(), cmap="gray")
        axes[i, 1].set_title("Reconstruction")
        axes[i, 2].imshow(y[i, 0].cpu(), cmap="gray")
        axes[i, 2].set_title("Partial Input")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

visualize_results(model, test_loader)
