import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

def loss_fun(recon_x, x, p, mask=None, ALPHA=1):
    if mask is None:
        mask = torch.ones(x.shape[0], x.shape[1])
    
    recon_x = recon_x[mask == 1]
    x = x[mask == 1]
    
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
                 embedding_dim=512,
                 temperature=1.0,
                 dropout=0.5,
                 loss_alpha=0,
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
        self.dropout = dropout
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
            enc_layers.append(nn.Dropout(self.dropout))
        
        enc_layers.append(nn.Conv1d(enc_chans[-1], num_tokens, kernel_size=3, padding=1))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder
        dec_chans = [hidden_dim] * num_layers
        dec_layers = []

        for in_chan, out_chan in zip(dec_chans[:-1], dec_chans[1:]):
            dec_layers.append(nn.Conv1d(in_chan, out_chan, kernel_size=3, padding=1))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Dropout(self.dropout))
        
        self.codebook = nn.Embedding(num_tokens, dec_chans[0])  # embedding layer converts one-hot vector into embedding vector
        
        dec_layers.append(nn.Conv1d(dec_chans[-1], input_chan, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x, mask=None):      
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1])   # if mask is not provided, mask out nothing.
            
        if len(x.shape) == 2:   # for unbatched data
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        x = x.permute(0, 2, 1) # (b, l, c) ----> (b, c, l)
        p = self.encoder(x)
        
        one_hot = F.gumbel_softmax(p, tau=self.temperature, hard=self.straight_through, dim=1) # reparametrization trick

        # if self.straight_through and self.reinmax:
        #     # use reinmax for better second-order accuracy - https://arxiv.org/abs/2304.08612
        #     # algorithm 2
        #     one_hot = one_hot.detach()
        #     π0 = p.softmax(dim = 1)
        #     π1 = (one_hot + (p / self.temperature).softmax(dim = 1)) / 2
        #     π1 = ((log(π1) - p).detach() + p).softmax(dim = 1)
        #     π2 = 2 * π1 - 0.5 * π0
        #     one_hot = π2 - π2.detach() + one_hot
        
        sampled = torch.einsum('bcl, nd -> bdl', one_hot, self.codebook.weight) # convert one-hot vector into embedding vector
        
        sampled = torch.einsum('bdl, bl -> bdl', sampled, mask) # mask out padding token
        
        recon_x = self.decoder(sampled)
        
        return recon_x.permute(0, 2, 1), p.permute(0, 2, 1)
    
    @torch.no_grad()
    def evaluate(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1])
            
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        x = x.permute(0, 2, 1)
        p = self.encoder(x)
        
        one_hot = F.one_hot(p.argmax(dim=1), num_classes=self.num_tokens).float()   # (b, l, c)
        one_hot = one_hot.permute(0, 2, 1) # (b, l, c) ----> (b, c, l)
        sampled = torch.einsum('bcl, nd -> bdl', one_hot, self.codebook.weight)
        
        sampled = torch.einsum('bdl, bl -> bdl', sampled, mask) # mask out padding token
        
        recon_x = self.decoder(sampled)
        
        return recon_x.permute(0, 2, 1), p.permute(0, 2, 1)
    
    def tokenize(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1])
            
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        
        tokens = x.argmax(dim=1)
        
        tokens += 1 # shift index by 1 to avoid 0 index for padding token
        tokens[mask == 0] = 0 # set padding token to 0
        
        return tokens
    
    def training_step(self, batch, batch_idx):
        x, mask = batch['x'], batch['mask']
        recon_x, p = self(x, mask)
        loss = loss_fun(recon_x, x, p, mask, ALPHA=self.loss_alpha)
        self.log('train_loss', loss, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask = batch['x'], batch['mask']
        recon_x, p = self.evaluate(x, mask)
        loss = loss_fun(recon_x, x, p, mask, ALPHA=self.loss_alpha)
        self.log('val_loss', loss, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}
                 

if __name__ == '__main__':
    model = GenomedVAE()
    print(model)
    x = torch.randn(4, 512, 320)
    mask = torch.ones(4, 512)
    mask[:, 100:] = 0
    
    # mask
    y, z = model(x, mask)
    y_eval, z_eval = model.evaluate(x, mask)
    
    embed_x  = model.tokenize(x, mask)
    
    mask_loss = loss_fun(y, x, z, mask)
    
    mask_val_loss = loss_fun(y_eval, x, z_eval, mask)
    
    # unmask
    y, z = model(x)
    y_eval, z_eval = model.evaluate(x)
    embed_x  = model.tokenize(x)
    loss = loss_fun(y, x, z)
    val_loss = loss_fun(y_eval, x, z_eval)
    