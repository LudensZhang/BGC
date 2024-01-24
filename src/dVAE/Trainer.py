import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dVAE.dVAE import GenomedVAE
from src.dVAE.Dataset import GenomedVAEDataset

def loss_fun(recon_x, x, p):
    MSE = F.mse_loss(recon_x, x)
    
    log_p = F.log_softmax(p, dim=2)
    log_uniform = torch.log(torch.tensor(1.0 / p.shape[-1]))
    KLD = F.kl_div(log_p, log_uniform, None, None, reduction='batchmean', log_target=True)
    
    return MSE + KLD

class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class dVAETrainer(object):
    def __init__(self, 
                 model, 
                 optimizer, 
                 epochs,
                 writer,
                 log_step,
                 eval_step,
                 early_stopping_patience,
                 save_path,
                 device):
        
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.writer = writer
        self.log_step = log_step
        self.eval_step = eval_step
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.device = device
        self.save_path = save_path
        self.loss_fun = loss_fun
        self.loggers = pd.DataFrame(columns=['steps', 'train_loss', 'val_loss'])
        
        
    def train(self, train_loader, val_loader):
        self.model.train()
        print(self.model)
        self.model.to(self.device)
        
        
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, data in enumerate(tqdm(train_loader)):
                recent_step = epoch*len(train_loader) + batch_idx
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, p = self.model(data)
                loss = self.loss_fun(recon_batch, data, p)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                
                if recent_step % self.log_step == 0:
                    self.writer.add_scalar('Loss/train', loss.item(), recent_step)
                    print(f'Step: {recent_step}/{self.epochs*len(train_loader)} Train Loss: {loss.item()}')
                    self.loggers = self.loggers._append({'steps': recent_step, 'train_loss': loss.item()}, ignore_index=True)

            
                if recent_step % self.eval_step == 0:
                    val_loss = self.eval(val_loader)
                    self.writer.add_scalar('Loss/val', val_loss, recent_step)
                    print(f'Step: {epoch*len(train_loader) + batch_idx}/{self.epochs*len(train_loader)} Val Loss: {val_loss}')
                    self.loggers = self.loggers._append({'steps': recent_step, 'val_loss': val_loss}, ignore_index=True)
                    
                    self.early_stopping(val_loss)
                    if self.early_stopping.early_stop:
                        print('Early stopping')
                        self.save(self.save_path)
                        break
            
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
            
    def eval(self, val_loader):
        self.model.eval()
        val_loss = 0
        print('Evaluating...')
        with torch.no_grad():
            for data in tqdm(val_loader):
                data = data.to(self.device)
                recon_batch, p = self.model(data)
                val_loss += self.loss_fun(recon_batch, data, p).item()
                
        val_loss /= len(val_loader.dataset)
        return val_loss
    
    def save(self):
        torch.save(self.model.state_dict(), self.save_path)
            
if __name__ == '__main__':
    train_data = GenomedVAEDataset(csv_file='train.csv')
    val_data = GenomedVAEDataset(csv_file='val.csv')
    
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=True)
    model = GenomedVAE()
    
    trainer = dVAETrainer(model=model,
                            optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5),
                            epochs=100,
                            writer=SummaryWriter('tmp_logs'),
                            log_step=100,
                            eval_step=500,
                            early_stopping_patience=10,
                            save_path='tmp_model/tmp.pth',
                            device=torch.device('cuda'))
    trainer.train(train_loader, val_loader)
    trainer.loggers.to_csv('tmp_logs/loggers.csv', index=False)