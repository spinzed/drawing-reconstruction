import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class CVAE(nn.Module):

    def __init__(self, input_dim, hidden_dim=400, latent_dim=200, K=10):
        super(CVAE, self).__init__()

        # encoder
        self.encoder = Encoder()
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, K)
        self.logvar_layer = nn.Linear(latent_dim, K)

        self.decoder = Decoder()
        
     
    def encode(self, x):
        z = self.encoder(x)
        mean, logvar = self.mean_layer(z), self.logvar_layer(z)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, z):
        x = self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def sample():
        epsilon = torch.randn_like(
        

class DNN_encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        return z
            
class CNN_encoder(nn.Module):
    def __init__(self, w, h, hidden_dim=400, latent_dim=200):
        self.cnn = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.LeakyRelu(0.2), 
            nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.LeakyRelu(0.2), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.LeakyRelu(0.2), 
            nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.LeakyRelu(0.2), 
            nn.MaxPool2d(2, 2),
            Flatten()
        )
            
        w_new = ((w - 8) // 2 - 4)
        h_new = ((h - 8) // 2 - 4)
        d = 64*w_new*h_new
            
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyRelu(0.2),
            nn.Linear(d, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyRelu(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward():
        
        
class DNN_decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        return z
            
class CNN_decoder(nn.Module):
    def __init__(self, w, h, hidden_dim=400, latent_dim=200):
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyRelu(0.2),
            nn.Linear(d, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.LeakyRelu(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
