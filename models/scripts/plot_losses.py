import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Config
# --------------------------------------------------
item = "ice cream"      # same item used during training
save_dir = "."          # directory where .npy files are saved

train_file = os.path.join(save_dir, f"{item}_train_loss.npy")
val_file = os.path.join(save_dir, f"{item}_val_loss.npy")

# --------------------------------------------------
# Load losses
# --------------------------------------------------
if not os.path.exists(train_file):
    raise FileNotFoundError(f"Train loss file not found: {train_file}")

if not os.path.exists(val_file):
    raise FileNotFoundError(f"Val loss file not found: {val_file}")

train_losses = np.load(train_file)   # shape (epochs, 3)
val_losses = np.load(val_file)       # shape (epochs, 3)

# --------------------------------------------------
# Split components
# --------------------------------------------------
train_total, train_recon, train_kl = train_losses.T
val_total, val_recon, val_kl = val_losses.T

epochs = np.arange(1, len(train_total) + 1)

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_total, label="Train Total", linewidth=2)
plt.plot(epochs, val_total, label="Val Total", linewidth=2)

plt.plot(epochs, train_recon, "--", label="Train Recon", alpha=0.7)
plt.plot(epochs, val_recon, "--", label="Val Recon", alpha=0.7)

plt.plot(epochs, train_kl, ":", label="Train KL", alpha=0.7)
plt.plot(epochs, val_kl, ":", label="Val KL", alpha=0.7)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"CVAE Training Losses ({item})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
