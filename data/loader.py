from matplotlib import pyplot as plt
import torch
import numpy as np
from models import cvae_model
from models import cvae_model_decoderz
from models import sketch_rnn_model
from data import utils
import os
import cv2

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

        if model_type == "CVAE":
            generator = CVaeGenerator()
        elif model_type == "CVAE_Decoderz":
            generator = CVaeDekoderzGenerator()
        elif model_type == "SketchRNN":
            generator = SketchRNNGenerator()
        else:
            print(f"unknown model type {model_type}")
            return None

        generator.init(model_type, checkpoint["supported_classes"], checkpoint)

        # for models that don't work on pixels like sketchrnn
        if generator.image_size is None:
            generator.image_size = self.image_size

        generator.set_weights(checkpoint)

        return generator

"""
Base implementation for image generators for each model type.
"""
class Generator():
    def init(self, model_type, supported_classes, checkpoint):
        self.model_type = model_type
        self.supported_classes = supported_classes
        self.weights_set = False
        self.image_size = None
        self.init_model(checkpoint)
        self.set_weights(checkpoint)
        self.checkpoint = checkpoint

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

class CVaeGenerator(Generator):
    def init_model(self, checkpoint):
        self.image_size = checkpoint["image_size"]
        self.model = cvae_model.CVAE(self.image_size[0], checkpoint["latent_dim"], device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["weights"])
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
            raise RuntimeError("weights must be set first")

        img_gen_size = utils.compile_img_from_strokes(strokes, img_shape=self.image_size, pad=0)
        img_gen_size = utils.erode_image(img_gen_size)
        img_gen_size = img_gen_size.astype(np.float32)

        plt.imshow(img_gen_size, cmap='gray')
        plt.show()

        self.model.eval()
        y = torch.tensor(img_gen_size, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            x_prob = self.model.sample(y)
        out = x_prob.squeeze(0).squeeze(0).cpu().detach().numpy()

        return out

    def additional_params(self):
        return {"image_size": self.image_size, "latent dim": self.model.latent_dim}

class CVaeDekoderzGenerator(Generator):
    def init_model(self, checkpoint):
        self.image_size = checkpoint["image_size"]
        self.model = cvae_model_decoderz.CVAE(self.image_size[0], checkpoint["latent_dim"], device=torch.device(device))
        self.model = self.model.to(device)

    def set_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["weights"])
        self.weights_set = True

    def generate(self, img, strokes):
        if not self.weights_set:
            raise RuntimeError("weights must be set first")

        img_gen_size = utils.compile_img_from_strokes(strokes, img_shape=self.image_size)
        #img_gen_size = cv2.resize(img, self.image_size, interpolation=cv2.INTER_NEAREST)
        img_gen_size = utils.erode_image(img_gen_size)
        img_gen_size = img_gen_size.astype(np.float32)

        self.model.eval()
        y = torch.tensor(img_gen_size, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        with torch.no_grad():
            x_prob = self.model.sample(y)
        out = x_prob.squeeze(0).squeeze(0).cpu().detach().numpy()

        return out

    def additional_params(self):
        return {"image_size": self.image_size, "latent dim": self.model.latent_dim}

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

        in_sequence = utils.strokes_to_relative_sequence(strokes, offset=(-self.image_size[1]//2, -self.image_size[0]//2))
        with torch.no_grad():
            out_sequence = self.model.sample(in_sequence)

        total_sequence = np.vstack([in_sequence, out_sequence])
        img_out = utils.compile_img_from_sequence(total_sequence, relative_offsets=True, img_shape=self.image_size, img=img)
        return img_out

    def additional_params(self):
        return {"Nmax": self.model.Nmax}
