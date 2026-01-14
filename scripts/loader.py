import torch
import vae_model
import vae_model_classic_bernoulli
import cvae_model
import cvae_model_decoderz
import sketch_rnn_model
import os
import utils as ut
import numpy as np

binarization_threshold = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
model_residual = False

def get_y(x, out):
    if model_residual:
        return torch.min(x, 1-out)
    return out

"""
Used by UI to load models from checkpoints.
"""

class Loader:
    def __init__(self, image_size):
        self.image_size = image_size

    def load_from_checkpoint(self, path):
        if not os.path.isfile(path):
            raise RuntimeError("checkpoint file doesn't exist")

        checkpoint = torch.load(path, weights_only=True)

        if type(checkpoint) is not dict:
            raise RuntimeError("checkpoint should be a dict")

        if "model_type" not in checkpoint:
            raise RuntimeError("cannot get model type from checkpoint")

        if "supported_classes" not in checkpoint:
            raise RuntimeError("cannot get supported classes from checkpoint")

        if len(checkpoint["supported_classes"]) == 0:
            raise RuntimeError("model apparently supports no classes?")

        model_type = checkpoint["model_type"]

        if model_type == "VAE":
            generator = VaeGenerator()
        elif model_type == "ConvVAE":
            generator = ConvVaeGenerator()
        elif model_type == "CVAE":
            generator = CVaeGenerator()
        elif model_type == "CVAE_Decoderz":
            generator = CVaeDekoderzGenerator()
        elif model_type == "SketchRNN":
            generator = SketchRNNGenerator()
        else:
            print(f"unknown model type {model_type}")
            return None

        generator.init(model_type, self.image_size, checkpoint["supported_classes"], checkpoint)
        generator.set_weights(checkpoint)

        return generator

"""
Base implementation for image generators for each model type.
"""
class Generator():
    def init(self, model_type, image_size, supported_classes, checkpoint):
        self.model_type = model_type
        self.image_size = image_size
        self.supported_classes = supported_classes
        self.weights_set = False
        self.init_model(checkpoint)
        self.set_weights(checkpoint)

    """
    Virtual methods to be implemented by each model type.
    """

    def init_model(self, checkpoint):
        pass

    def set_weights(self, checkpoint):
        pass

    # model can use either img or strokes (or both)
    def generate(self, img, strokes):
        pass

    def additional_params(self):
        return dict()

class VaeGenerator(Generator):
    def init_model(self, checkpoint):
        self.weights_file = None
        self.model = vae_model.VAE(self.image_size[0] * self.image_size[1], checkpoint["latent_dim"], checkpoint["hidden_dim"])
        self.model = self.model.to(device)

    def set_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["weights"])
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        imgd = torch.tensor(img).to(device)
        X = imgd.reshape((1, self.image_size[0] * self.image_size[1]))
        out = self.model(X)
        return get_y(img, out)[0].cpu().detach().numpy()

    def additional_params(self):
        return {"latent dim": self.model.latent_dim, "hidden dim": self.model.hidden_dim}

class ConvVaeGenerator(Generator):
    def init_model(self, checkpoint):
        self.weights_file = None
        self.model = vae_model_classic_bernoulli.VAE(self.image_size[0], checkpoint["latent_dim"], device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["weights"])
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
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

    def additional_params(self):
        return {"latent dim": self.model.latent_dim}

class CVaeGenerator(Generator):
    def init_model(self, checkpoint):
        self.model = cvae_model.CVAE(self.image_size[0], checkpoint["latent_dim"], device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["weights"])
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        y = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            x_prob = self.model.sample(y)
        out = x_prob.squeeze(0).squeeze(0).cpu().detach().numpy()
        return out

    def additional_params(self):
        return {"latent dim": self.model.latent_dim}

class CVaeDekoderzGenerator(Generator):
    def init_model(self, checkpoint):
        self.model = cvae_model_decoderz.CVAE(self.image_size[0], checkpoint["latent_dim"], device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["weights"])
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
            raise RuntimeError("weights must be set first")

        self.model.eval()
        y = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            x_prob = self.model.sample(y)
        out = x_prob.squeeze(0).squeeze(0).cpu().detach().numpy()
        return out

    def additional_params(self):
        return {"latent dim": self.model.latent_dim}

class SketchRNNGenerator(Generator):
    def init_model(self, checkpoint):
        self.model = sketch_rnn_model.SketchRNN(0)
        self.model.load_settings(checkpoint)

    def set_weights(self, checkpoint):
        self.model.load_weights(checkpoint)
        self.model.encoder.to(device)
        self.model.decoder.to(device)
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
            raise RuntimeError("weights must be set first")

        self.model.encoder.eval()
        self.model.decoder.eval()
        in_sequence = ut.strokes_to_relative_sequence(strokes)
        with torch.no_grad():
            out_sequence = self.model.sample(in_sequence)
        #print("in_seq", in_sequence)
        #print("out_seq", out_sequence)
        total_sequence = np.vstack([in_sequence, out_sequence])
        img_out = ut.compile_img_from_sequence(total_sequence, relative_offsets=True, img_shape=self.image_size, img=img)
        return img_out

    def additional_params(self):
        return {"Nmax": self.model.Nmax}
