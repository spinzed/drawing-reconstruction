import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cvae_model import CVAE as VAE
from data.dataset import ImageDataset

# --------------------------
# Configuration
# --------------------------
dataset_dir = "quickdraw"
image_size = (64, 64)
item = "ice cream"
latent_dim = 32
model_weights_path = "weights/ice_cream.pth_n70000_epoch45.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
n_samples = 64  # 8x8 grid
image_index = 0  # index of the partial image to condition on
binarization_threshold = 0.5  # pixels below this become 0

# --------------------------
# Helper function
# --------------------------
def binarize(img, threshold=binarization_threshold):
    binary = np.ones_like(img, dtype=np.float32)
    binary[img < threshold] = 0
    return binary

# --------------------------
# Load dataset
# --------------------------
dataset = ImageDataset(dataset_dir, image_size, image_limit=10, item=item)
img_full, img_partial, cat = dataset[image_index]

# Preprocess
y = torch.tensor(img_partial, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# --------------------------
# Load model
# --------------------------
model = VAE(image_size[0], latent_dim, device)
checkpoint = torch.load(model_weights_path, map_location=device)
model.load_state_dict(checkpoint["weights"])
model.to(device)
model.eval()

# --------------------------
# Generate binarized samples
# --------------------------
samples = []
with torch.no_grad():
    for _ in range(n_samples):
        sample = model.sample(y)  # shape: (1,1,H,W)
        sample = sample.squeeze(0).squeeze(0).cpu().numpy()
        sample = binarize(sample)  # binarize here
        samples.append(sample)

# --------------------------
# Plotting in 8x8 grid
# --------------------------
grid_size = 8
plt.figure(figsize=(12, 12))
for i, s in enumerate(samples[:grid_size * grid_size]):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(s, cmap="Greys_r")
    plt.axis("off")

plt.tight_layout()
plt.show()
