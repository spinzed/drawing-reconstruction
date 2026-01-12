import torch
import vae_model
import vae_model_classic_bernoulli
import cvae_model
import cvae_model_decoderz
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
image_size = (256, 256)
model_residual = False

def get_y(x, out):
    if model_residual:
        return torch.min(x, 1-out)
    return out

"""
Used by UI to generate an image.
"""
class VaeGenerator():
    def __init__(self):
        self.weights_file = None
        self.model = vae_model.VAE(image_size[0] * image_size[1], 800, 400)
        self.model = self.model.to(device)

    def set_weights(self, path):
        if not os.path.isfile(path):
            raise RuntimeError("weights file doesn't exist")

        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.weights_file = path

    def generate(self, img):
        if self.weights_file is None:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        imgd = torch.tensor(img).to(device)
        X = imgd.reshape((1, image_size[0] * image_size[1]))
        out = self.model(X)
        return get_y(img, out)[0].cpu().detach().numpy()

class ConvVaeGenerator():
    def __init__(self):
        self.weights_file = None
        self.model = vae_model_classic_bernoulli.VAE(image_size[0], 200, device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, path):
        if not os.path.isfile(path):
            raise RuntimeError("weights file doesn't exist")

        state = torch.load(path, map_location=torch.device(device))
        self.model.load_state_dict(state)
        self.weights_file = path

    def generate(self, img):
        if self.weights_file is None:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        imgd = torch.tensor(img, dtype=torch.float32, device=device)
        X = imgd.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            enc_mean, enc_logvar, dec_mean, _ = self.model(X)
            out_probs = torch.sigmoid(dec_mean)
            out_t = get_y(X, out_probs)
        out = out_t.squeeze(0).squeeze(0).cpu().detach().numpy()
        return out

class CVaeGenerator():
    def __init__(self):
        self.weights_file = None
        self.model = cvae_model.CVAE(image_size[0], 64, device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, path):
        if not os.path.isfile(path):
            raise RuntimeError("weights file doesn't exist")

        state = torch.load(path, map_location=torch.device(device))
        self.model.load_state_dict(state)
        self.weights_file = path

    def generate(self, img):
        if self.weights_file is None:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        y = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            x_prob = self.model.sample(y)
        out = x_prob.squeeze(0).squeeze(0).cpu().detach().numpy()
        return out

class CVaeDekoderzGenerator():
    def __init__(self):
        self.weights_file = None
        self.model = cvae_model_decoderz.CVAE(image_size[0], 128, device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, path):
        if not os.path.isfile(path):
            raise RuntimeError("weights file doesn't exist")

        state = torch.load(path, map_location=torch.device(device))
        self.model.load_state_dict(state)
        self.weights_file = path

    def generate(self, img):
        if self.weights_file is None:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        y = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            x_prob = self.model.sample(y)
        out = x_prob.squeeze(0).squeeze(0).cpu().detach().numpy()
        print(out)
        return out

