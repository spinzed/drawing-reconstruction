import torch
import torch.nn as nn
from torch.distributions.normal import Normal

device = "cuda" if torch.cuda.is_available() else "cpu"


class GaussianEncoder(nn.Module):
    """Convolutional encoder producing (mu, logvar)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )

        # Automatically compute flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim, input_dim)
            h = self.net(dummy)
            self.flatten_dim = h.view(1, -1).size(1)

        self.mean = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)  # keep batch dimension
        return self.mean(h), self.logvar(h)


class GaussianDecoder(nn.Module):
    """Convolutional decoder modeling p(x|z)"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.init_channels = 32
        self.init_spatial = output_dim // 4  # must match encoder downsampling

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.init_channels * self.init_spatial * self.init_spatial),
            nn.LeakyReLU(0.2)
        )

        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

    
        self.mean_head = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        self.logvar_head = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.init_channels, self.init_spatial, self.init_spatial)
        h = self.trunk(h)

        x_mean = self.mean_head(h)
        x_logvar = self.logvar_head(h)
        return x_mean, x_logvar


class VAE(nn.Module):
    """Minimal VAE keeping per-pixel logvar loss"""
    def __init__(self, input_dim, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = GaussianEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = GaussianDecoder(latent_dim=latent_dim, output_dim=input_dim)


    def forward(self, x):
        enc_mean, enc_logvar = self.encoder(x)
        z = reparametrization(enc_mean, enc_logvar)
        dec_mean, dec_logvar = self.decoder(z)
        return enc_mean, enc_logvar, dec_mean, dec_logvar

    def sample(self, temperature = 1):
        z = torch.randn([1, self.latent_dim])
        dec_mean, dec_logvar = self.decoder(z)
        return reparametrization(dec_mean, dec_logvar * temperature)
    
    @staticmethod
    def loss(x, enc_mean, enc_logvar, dec_mean, dec_logvar):
        z = reparametrization(enc_mean, enc_logvar)

        dec_dist = Normal(dec_mean, torch.exp(0.5 * dec_logvar))
        enc_dist = Normal(enc_mean, torch.exp(0.5 * enc_logvar))
        standard_dist = Normal(torch.zeros_like(z), torch.ones_like(z))

        neglog_dec = -dec_dist.log_prob(x)
        neglog_dec = torch.flatten(neglog_dec, start_dim=1).sum(dim=-1)
        neglog_enc = -enc_dist.log_prob(z).sum(dim=-1)
        neglog_pz = -standard_dist.log_prob(z).sum(dim=-1)
        loss = (neglog_dec - neglog_enc + neglog_pz).mean(dim=0)
        return loss

def reparametrization(mean, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mean + std * epsilon
