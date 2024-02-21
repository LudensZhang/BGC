import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dVAE.dVAE import GenomedVAE
from src.dVAE.Trainer import dVAETrainer
from src.dVAE.Dataset import GenomedVAEDataset
from pickle import dump, load
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):  # DataLoaderX for faster reading speed.
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def main():
    # train_data = GenomedVAEDataset(csv_file='train.csv')
    # dump(train_data, open('data/train_data.pkl', 'wb'))
    # val_data = GenomedVAEDataset(csv_file='val.csv')
    # dump(val_data, open('data/val_data.pkl', 'wb'))
    
    train_data = load(open('data/train_data.pkl', 'rb'))
    val_data = load(open('data/val_data.pkl', 'rb'))
    
    train_loader = DataLoaderX(train_data, batch_size=1024, shuffle=True)
    val_loader = DataLoaderX(val_data, batch_size=1024, shuffle=True)
    model = GenomedVAE()

    trainer = dVAETrainer(model=model,
                            optimizer=optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5),
                            epochs=100,
                            writer=SummaryWriter('dVAE_logs'),
                            log_step=100,
                            eval_step=500,
                            early_stopping_patience=5,
                            save_path='dVAE_model/0123.pth',
                            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    trainer.train(train_loader, val_loader)
    
def lightning_main():
    train_data = GenomedVAEDataset(csv_file='train.csv')
    val_data = GenomedVAEDataset(csv_file='val.csv')
        
    train_loader = DataLoaderX(train_data, 
                               batch_size=16, 
                               shuffle=True)
    val_loader = DataLoaderX(val_data, batch_size=128)

    model = GenomedVAE(num_tokens=2048, 
                       hidden_dim=512,
                       temperature=0.9)
    early_stopping = EarlyStopping('val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=50, save_top_k=-1)
    
    logger = CSVLogger('lightning_logs', name=f'dVAE_{model.num_tokens}')
    logger.log_hyperparams(model.hparams)
    trainer = Trainer(accelerator='auto',
                      max_epochs=1, 
                      logger=logger, 
                      callbacks=[checkpoint_callback, early_stopping],
                      val_check_interval=0.5,
                      log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f'dVAE_model/dVAE_{model.num_tokens}_{model.temperature}.ckpt')
    
def pre_exp():
    
    # import pandas as pd
    # pd.read_csv('train.csv').sample(100).to_csv('train_100.csv', index=False)
    # train_data = GenomedVAEDataset(csv_file='train_100.csv')
    # train_data = dump(train_data, open('data/train_data_100.pkl', 'wb'))
    # pd.read_csv('val.csv').sample(100).to_csv('val_100.csv', index=False)
    # val_data = GenomedVAEDataset(csv_file='val_100.csv')
    # val_data = dump(val_data, open('data/val_data_100.pkl', 'wb'))
    
    train_data = load(open('data/train_data_100.pkl', 'rb'))
    val_data = load(open('data/val_data_100.pkl', 'rb'))
    
    
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)
    
    model = GenomedVAE(temperature=0.5)
    early_stopping = EarlyStopping('val_loss', patience=5)
    logger = CSVLogger('lightning_logs', name='dVAE_test')
    logger.log_hyperparams(model.hparams)
    
    trainer = Trainer(accelerator='auto',
                        max_epochs=100, 
                        logger=logger, 
                        callbacks=[early_stopping], 
                        val_check_interval=0.5,
                        log_every_n_steps=100)
    
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint('dVAE_model/test.ckpt')
    
    model = GenomedVAE.load_from_checkpoint('dVAE_model/test.ckpt', map_location=torch.device('cpu'))
    x = val_data[:5]
    y, z = model(x)
    y_eval, z_eval = model.evaluate(x)
    embed_x  = model.tokenize(x)
    
    
def batch_size_select():
    import panas as pd
    import time
    
    train_data = GenomedVAEDataset(csv_file='train.csv')
    
    model = GenomedVAE()
    
    for batch_size in range(1, 1000, 10):
        start = time.time()
        batch = train_data[:batch_size]
        end = time.time()
        read_time = end - start
        
        start = time.time()
        model.training_step(batch, 0)
        end = time.time()
        train_time = end - start
        
        print(f'batch_size: {batch_size}, read_time: {read_time}, train_time: {train_time}, read_time - train_time: {read_time - train_time}')

if __name__ == '__main__':
    # main()
    lightning_main()
    # pre_exp()
    
    