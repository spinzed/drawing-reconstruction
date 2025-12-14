import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Classic VAE as proposed by Kingston
"""


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=400, latent_dim=200, path_size = 30):
        super().__init__()
        self.encoder = GaussianEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        self.decoder = GaussianDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim
        ) 
        
    def reparametrization(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std).to(device)      
        z = mean + std*epsilon
        return

    def sample(size):
        z = torch.randn([size, self.latent_dim])
        

    def forward(self, x):
        enc_mean, enc_logvar = self.encoder(x)
        z = self.reparametrization(enc_mean, enc_logvar)
        dec_mean, dec_logvar = self.decoder(z)
        x_hat = self.decode(z)
        return x_hat, enc_mean, enc_logvar, dec_mean, dec_logvar

    def loss(self, x):
        """
        Implementation of elbo for vae
        """
        x, dec_mean, dec_logvar, enc_mean, enc_logvar = self.forward(x)

        z = reparametrization(enc_mean, enc_logvar)
        
        dec_dist = Normal(dec_mean, torch.exp(0.5*dec_logvar))
        enc_dist = Normal(enc_mean, torch.exp(0.5*enc_logvar))
        standard_dist = Normal(torch.zeros_like(z), torch.ones_like(z))
        
        neglog_dec = -dec_dist.log_prob(z)
        neglog_enc = -enc_dist.log_prob(x)
        neglog_pz = -standard_dist.log_prob(z)

        loss = (neglog_dec - neglog_enc + neglog_pz).mean()

        return loss


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

