import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

device = "cuda" if torch.cuda.is_available() else "cpu"

class IPA(nn.Module):
    """
    CVAE inspired by https://arxiv.org/pdf/2102.12037
    """

    def __init__(self, input_dim, hidden_dim=400, latent_dim=200, ):
        super().__init__()
        self.hidden_dim = 400
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.full_encoder = FullEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        self.partial_encoder = PartialEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        self.decoder = GaussianDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim
        )
    
    def load_weights_from_pretrained_VAE(self, vae_model):
        if self.latent_dim == vae_model.latent_dim and \
            self.input_dim == vae_model.input_dim and \
            self.hidden_dim == self.hidden_dim:
            self.full_encoder = vae_model.encoder
            self.decoder = vae_model.decoder
        else: raise ValueError("vae and cvae weighst do not correspon to each other")

    def freeze_full_encoder(self, Value: bool):
        for param in self.full_encoder.parameters():
            param.requires_grad = not Value
        

    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x_full, x_part):
        mu_f, logvar_f = self.full_encoder(x_full)
        mu_p, logvar_p = self.partial_encoder(x_part)

        z = self.reparameterize(mu_f, logvar_f)

        x_mu, x_logvar = self.decoder(z)

        return x_mu, x_logvar, mu_f, logvar_f, mu_p, logvar_p

    def loss(self, x, x_full, x_part):
        """
        ELBO for IPA CVAE
        """
        x_mu, x_logvar, mu_f, logvar_f, mu_p, logvar_p = self.forward(x_full, x_part)

        recon_dist = Normal(x_mu, torch.exp(0.5 * x_logvar))
        recon_loss = -recon_dist.log_prob(x).sum(dim=-1)

        q_full = Normal(mu_f, torch.exp(0.5 * logvar_f))
        q_part = Normal(mu_p, torch.exp(0.5 * logvar_p))
        kl = kl_divergence(q_full, q_part).sum(dim=-1)

        return (recon_loss + kl).mean()


class GaussianEncoder(nn.Module):
    """
    Generic encoder producing (mu, logvar)
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.logvar(h)


class FullEncoder(GaussianEncoder):
    pass


class PartialEncoder(GaussianEncoder):
    pass


class GaussianDecoder(nn.Module):
    """
    Decoder modeling p(x|z)
    """
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.net(z)
        return self.mean(h), self.logvar(h)

