import numpy as np
import matplotlib.pyplot as plt
import PIL
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
class_name = "alarm clock"

###################################### hyperparameters
class HParams():
    def __init__(self):
        self.data_location = f"quickdraw/{class_name}.npz"
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512
        self.Nz = 128
        self.M = 20
        self.dropout = 0.9
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200

hp = HParams()

################################# load and prepare data
def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= hp.max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data, scale_factor

def rescale(strokes, scale):
    """Rescale entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    for seq in strokes:
        seq[:, 0:2] *= scale
        data.append(seq)
    return data

############################## function to generate a batch:
def make_batch(data, batch_size, Nmax):
    batch_idx = np.random.choice(len(data),batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:,0])
        new_seq = np.zeros((Nmax,5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0
        lengths.append(len(seq[:,0]))
        strokes.append(new_seq)
        indice += 1

    batch = torch.from_numpy(np.stack(strokes,1)).float()
    if use_cuda:
        batch = batch.cuda()
    return batch, lengths

# pads from (N, 3) to (Nmax, 5)
def transform_sequence(sequence, Nmax):
    len_seq = len(sequence[:, 0])
    new_seq = np.zeros((Nmax, 5))
    new_seq[:len_seq, :3] = sequence[:, :3]
    new_seq[:len_seq-1, 3] = 1 - sequence[:len_seq-1, 2]
    new_seq[(len_seq):, 4] = 1
    return new_seq

################################ adaptive lr
def lr_decay(optimizer):
    """Decay learning rate by a factor of lr_decay"""
    for param_group in optimizer.param_groups:
        if param_group["lr"]>hp.min_lr:
            param_group["lr"] *= hp.lr_decay
    return optimizer

################################# encoder and decoder modules
class EncoderRNN(nn.Module):
    def __init__(self, Nmax):
        super(EncoderRNN, self).__init__()
        self.Nmax = Nmax

        # bidirectional lstm (no dropout when num_layers is 1 to avoid torch warning):
        self.lstm = nn.LSTM(5, hp.enc_hidden_size, dropout=0, bidirectional=True)
        # create mu and sigma from lstm"s last output:
        self.fc_mu = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        self.fc_sigma = nn.Linear(2*hp.enc_hidden_size, hp.Nz)
        # active dropout:
        self.train()

    def forward(self, inputs, batch_size, hidden_cell=None):
        if hidden_cell is None:
            # then must init with zeros
            if use_cuda:
                hidden = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
                cell = torch.zeros(2, batch_size, hp.enc_hidden_size).cuda()
            else:
                hidden = torch.zeros(2, batch_size, hp.enc_hidden_size)
                cell = torch.zeros(2, batch_size, hp.enc_hidden_size)
            hidden_cell = (hidden, cell)
        _, (hidden,cell) = self.lstm(inputs.float(), hidden_cell)
        # hidden is (2, batch_size, hidden_size), we want (batch_size, 2*hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden,1,0)
        hidden_cat = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)],1)
        # mu and sigma:
        mu = self.fc_mu(hidden_cat)
        sigma_hat = self.fc_sigma(hidden_cat)
        sigma = torch.exp(sigma_hat/2.)
        # N ~ N(0,1)
        z_size = mu.size()
        if use_cuda:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size)).cuda()
        else:
            N = torch.normal(torch.zeros(z_size),torch.ones(z_size))
        z = mu + sigma*N
        # mu and sigma_hat are needed for LKL loss
        return z, mu, sigma_hat

class DecoderRNN(nn.Module):
    def __init__(self, Nmax):
        super(DecoderRNN, self).__init__()
        self.Nmax = Nmax
        # to init hidden and cell from z:
        self.fc_hc = nn.Linear(hp.Nz, 2*hp.dec_hidden_size)
        # unidirectional lstm (no dropout when num_layers is 1 to avoid torch warning):
        self.lstm = nn.LSTM(hp.Nz+5, hp.dec_hidden_size, dropout=0)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size,6*hp.M+3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden,cell = torch.split(F.tanh(self.fc_hc(z)),hp.dec_hidden_size,1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs,(hidden,cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the lstm with the whole input in one shot
        # and use all outputs contained in "outputs", while in generate
        # mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y,6,1)
        params_mixture = torch.stack(params[:-1]) # trajectory
        params_pen = params[-1] # pen up/down
        # identify mixture params:
        pi,mu_x,mu_y,sigma_x,sigma_y,rho_xy = torch.split(params_mixture,1,2)
        # preprocess params::
        if self.training:
            len_out = self.Nmax+1
        else:
            len_out = 1

        pi = F.softmax(pi.transpose(0,1).squeeze(), -1).view(len_out,-1,hp.M)
        sigma_x = torch.exp(sigma_x.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        sigma_y = torch.exp(sigma_y.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        rho_xy = torch.tanh(rho_xy.transpose(0,1).squeeze()).view(len_out,-1,hp.M)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out,-1,hp.M)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out,-1,hp.M)
        q = F.softmax(params_pen, -1).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell

class SketchRNN():
    def __init__(self, Nmax):
        self.encoder = EncoderRNN(Nmax).to(device)
        self.decoder = DecoderRNN(Nmax).to(device)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min
        self.Nmax = Nmax

    def set_data(self, data):
        self.data = data

    def make_target(self, batch, lengths):
        if use_cuda:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.size()[1]).cuda().unsqueeze(0)
        else:
            eos = torch.stack([torch.Tensor([0,0,0,0,1])]*batch.size()[1]).unsqueeze(0)
        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(self.Nmax+1, batch.size()[1])
        for indice,length in enumerate(lengths):
            mask[:length,indice] = 1
        if use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch[:, :, 0]]*hp.M,2)
        dy = torch.stack([batch[:, :, 1]]*hp.M,2)
        p1 = batch[:, :, 2]
        p2 = batch[:, :, 3]
        p3 = batch[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)
        return mask, dx, dy, p

    def train(self, epoch):
        self.encoder.train()
        self.decoder.train()
        batch, lengths = make_batch(self.data, hp.batch_size, self.Nmax)
        # encode:
        z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)
        # create start of sequence:
        if use_cuda:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).cuda().unsqueeze(0)
        else:
            sos = torch.stack([torch.Tensor([0,0,1,0,0])]*hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch],0)
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z]*(self.Nmax+1))
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack],2)
        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, _, _ = self.decoder(inputs, z)
        # prepare targets:
        mask,dx,dy,p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        self.eta_step = 1-(1-hp.eta_min)*hp.R
        # compute losses:
        LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask,dx,dy,p,epoch)
        loss = LR + LKL
        # gradient step
        loss.backward()
        # gradient cliping
        nn.utils.clip_grad_norm(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm(self.decoder.parameters(), hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # some print and save:
        if epoch%1==0:
            print("epoch", epoch, "loss", loss.item(), "LR", LR.item(), "LKL", LKL.item())
            self.encoder_optimizer = lr_decay(self.encoder_optimizer)
            self.decoder_optimizer = lr_decay(self.decoder_optimizer)
        if epoch%100==0:
            #self.save(epoch)
            self.conditional_generation(epoch)
            pass

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx-self.mu_x)/self.sigma_x)**2
        z_y = ((dy-self.mu_y)/self.sigma_y)**2
        z_xy = (dx-self.mu_x)*(dy-self.mu_y)/(self.sigma_x*self.sigma_y)
        z = z_x + z_y -2*self.rho_xy*z_xy
        exp = torch.exp(-z/(2*(1-self.rho_xy**2)))
        norm = 2*np.pi*self.sigma_x*self.sigma_y*torch.sqrt(1-self.rho_xy**2)
        return exp/norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)
        LS = -torch.sum(mask*torch.log(1e-5+torch.sum(self.pi * pdf, 2)))\
            /float(self.Nmax*hp.batch_size)
        LP = -torch.sum(p*torch.log(self.q))/float(self.Nmax*hp.batch_size)
        return LS+LP

    def kullback_leibler_loss(self):
        LKL = -0.5*torch.sum(1+self.sigma-self.mu**2-torch.exp(self.sigma))\
            /float(hp.Nz*hp.batch_size)
        KL_min = torch.tensor([hp.KL_min], dtype=torch.float32, device=self.mu.device).detach()
        return hp.wKL*self.eta_step * torch.max(LKL,KL_min)

    def save(self, epoch=None, filename=None):
        if filename is None:
            filename = "weights/weights_sketchrnn.pth"

        checkpoint = {
            "model_type": "SketchRNN",
            "supported_classes": [class_name],
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "encoder_optimizer_state": getattr(self, "encoder_optimizer", None) and self.encoder_optimizer.state_dict(),
            "decoder_optimizer_state": getattr(self, "decoder_optimizer", None) and self.decoder_optimizer.state_dict(),
            "hp": hp.__dict__,
            "epoch": epoch,
            "nmax": self.Nmax,
        }
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")

    def load_settings(self, checkpoint):
        self.Nmax = checkpoint.get("nmax", self.Nmax)

    def load_weights(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        if "encoder_optimizer_state" in checkpoint and checkpoint["encoder_optimizer_state"] is not None:
            self.encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state"])
            self.decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state"])

    def conditional_generation(self, epoch):
        batch, lengths = make_batch(self.data, 1, self.Nmax)

        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)

        # encode:
        z, _, _ = self.encoder(batch, 1)
        sos = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).view(1,1,-1)
        if use_cuda:
            sos = sos.cuda()
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.Nmax):
            input = torch.cat([s, z.unsqueeze(0)],2)
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            #------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print("Eos:", i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence, epoch)

    def sample(self, y: np.ndarray):
        if len(y) > self.Nmax:
            print(f"input too long (gotten {len(y)} lines, but max allowed is {self.Nmax}), clamping...")
            y = y[:self.Nmax, :]

        Y, scale_factor = normalize([y.astype(np.float32)])
        Y = Y[0]
        y = transform_sequence(y, self.Nmax)
        y = torch.from_numpy(np.stack(np.array([y]), 1)).float().to(device)

        # encode:
        z, _, _ = self.encoder(y, 1)
        sos = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32).view(1, 1, -1).to(device)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(self.Nmax):
            input = torch.cat([s, z.unsqueeze(0)], 2)

            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)

            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()

            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)

            if eos:
                print(i)
                break

        #x_sample = np.cumsum(seq_x, 0)
        #y_sample = np.cumsum(seq_y, 0)
        #z_sample = np.array(seq_z)
        x_sample = np.array(seq_x)
        y_sample = np.array(seq_y)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        sequence_rescaled = rescale([sequence], scale_factor)[0]
        return sequence_rescaled.astype(np.int32)

    def sample_next_state(self):

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(pi_pdf)/hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi[0,0,:].detach().cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        # get pen state:
        q = self.q[0,0,:].detach().cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x[0,0,pi_idx].item()
        mu_y = self.mu_y[0,0,pi_idx].item()
        sigma_x = self.sigma_x[0,0,pi_idx].item()
        sigma_y = self.sigma_y[0,0,pi_idx].item()
        rho_xy = self.rho_xy[0,0,pi_idx].item()
        x,y = sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy,greedy=False)
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx+2] = 1
        next_state = next_state.view(1,1,-1)
        if use_cuda:
            next_state = next_state.cuda()
        return next_state, x, y, q_idx==1, q_idx==2

def sample_bivariate_normal(mu_x,mu_y,sigma_x,sigma_y,rho_xy, greedy=False):

    # inputs must be floats
    if greedy:
      return mu_x,mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)
    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],\
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

def make_image(sequence, epoch, name="_output_"):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    fig = plt.figure()
    _ = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    # Robustly get RGB buffer from different backends
    try:
        raw = canvas.tostring_rgb()
        pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), raw)
    except AttributeError:
        # Fallback: use print_to_buffer which returns ARGB buffer
        buf, (w, h) = canvas.print_to_buffer()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        rgb = arr[:, :, 1:4]
        pil_image = PIL.Image.fromarray(rgb, "RGB")
    if not os.path.exists("output/"):
        os.makedirs("output")
    name = "output/" + str(epoch) + name+ ".jpg"
    pil_image.save(name, "JPEG")
    plt.close("all")

    #smin = np.min(np.abs(sequence[:, 0:2]))
    #smax = np.max(np.abs(sequence[:, 0:2]))

    #s2 = sequence
    #s2[:, 0:2] = ((s2[:, 0:2])/(smax-smin)) * 128 + 128
    #s2 = s2.astype(np.int32)
    #img = ut.compile_img_from_sequence(s2, relative_offsets=False)
    #plt.imshow(img, cmap="gray")
    #plt.show()

if __name__=="__main__":
    dataset = np.load(hp.data_location, encoding="latin1", allow_pickle=True)
    data = dataset["train"]
    data = purify(data)
    data, _ = normalize(data)
    Nmax = max_size(data)

    model = SketchRNN(Nmax)
    model.set_data(data)
    try:
        for epoch in range(50001):
            model.train(epoch)
    except KeyboardInterrupt:
        print("Training stopped early")

    model.save()

    """
    model.load("encoder.pth","decoder.pth")
    model.conditional_generation(0)
    #"""
