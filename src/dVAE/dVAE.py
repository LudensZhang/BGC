import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

def loss_fun(recon_x, x, p, ALPHA=1.0):
    MSE = F.mse_loss(recon_x, x)
    
    log_p = F.log_softmax(p, dim=2)
    log_uniform = torch.log(torch.tensor(1.0 / p.shape[-1]))
    KLD = F.kl_div(log_p, log_uniform, None, None, reduction='batchmean', log_target=True)
    
    return MSE + ALPHA * KLD

def log(t, eps=1e-20):
    return torch.log(t.clamp(min = eps))

class GenomedVAE(LightningModule):
    def __init__(self,
                 genome_length=512,
                 input_chan=320,
                 num_layers=3,
                 hidden_dim=64,
                 num_tokens=1024,
                 embedding_dim=64,
                 temperature=1.0,
                 loss_alpha=1.0,
                 straight_through=True,
                 reinmax=True):
        super(GenomedVAE, self).__init__()
        self.save_hyperparameters()
        self.genome_length = genome_length
        self.input_chan = input_chan
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.loss_alpha = loss_alpha
        
        self.straight_through = straight_through
        self.reinmax = reinmax
        
        # Encoder
        enc_chans = [hidden_dim] * num_layers
        enc_chans.insert(0, input_chan)
        enc_layers = []
        
        for in_chan, out_chan in zip(enc_chans[:-1], enc_chans[1:]):
            enc_layers.append(nn.Conv1d(in_chan, out_chan, kernel_size=3, padding=1))
            enc_layers.append(nn.ReLU())
        
        enc_layers.append(nn.Conv1d(enc_chans[-1], num_tokens, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_chans = [hidden_dim] * num_layers
        dec_layers = []

        for in_chan, out_chan in zip(dec_chans[:-1], dec_chans[1:]):
            dec_layers.append(nn.Conv1d(in_chan, out_chan, kernel_size=3, padding=1))
            dec_layers.append(nn.ReLU())
        
        self.codebook = nn.Embedding(num_tokens, dec_chans[0])  # embedding layer converts one-hot vector into embedding vector
        
        dec_layers.append(nn.Conv1d(dec_chans[-1], input_chan, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x):
        if len(x.shape) == 2:   # for unbatched data
            x = x.unsqueeze(0)
            
        x = x.permute(0, 2, 1) # (b, l, c) ----> (b, c, l)
        p = self.encoder(x)
        
        one_hot = F.gumbel_softmax(p, tau=self.temperature, hard=self.straight_through, dim=1) # reparametrization trick

        if self.straight_through and self.reinmax:
            # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
            # algorithm 2
            one_hot = one_hot.detach()
            π0 = p.softmax(dim = 1)
            π1 = (one_hot + (p / self.temperature).softmax(dim = 1)) / 2
            π1 = ((log(π1) - p).detach() + p).softmax(dim = 1)
            π2 = 2 * π1 - 0.5 * π0
            one_hot = π2 - π2.detach() + one_hot
        
        sampled = torch.einsum('bcl, nd -> bdl', one_hot, self.codebook.weight) # convert one-hot vector into embedding vector
        
        recon_x = self.decoder(sampled)
        
        return recon_x.permute(0, 2, 1), p.permute(0, 2, 1)
    
    @torch.no_grad()
    def evaluate(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        x = x.permute(0, 2, 1)
        p = self.encoder(x)
        
        one_hot = F.one_hot(p.argmax(dim=1), num_classes=self.num_tokens).float()   # (b, l, c)
        one_hot = one_hot.permute(0, 2, 1) # (b, l, c) ----> (b, c, l)
        sampled = torch.einsum('bcl, nd -> bdl', one_hot, self.codebook.weight)
        
        recon_x = self.decoder(sampled)
        
        return recon_x.permute(0, 2, 1), p.permute(0, 2, 1)
    
    def tokenize(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        tokens = x.argmax(dim=1)
        
        return tokens
    
    def training_step(self, batch, batch_idx):
        x = batch
        recon_x, p = self(x)
        loss = loss_fun(recon_x, x, p, ALPHA=self.loss_alpha)
        self.log('train_loss', loss, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        recon_x, p = self.evaluate(x)
        loss = loss_fun(recon_x, x, p, ALPHA=self.loss_alpha)
        self.log('val_loss', loss, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)       
                 

if __name__ == '__main__':
    model = GenomedVAE()
    print(model)
    x = torch.randn(4, 512, 320)
    y, z = model(x)
    y_eval, z_eval = model.evaluate(x)
    
    embed_x  = model.tokenize(x)
    
    loss = loss_fun(y, x, z)
    val_loss = loss_fun(y_eval, x, z_eval)