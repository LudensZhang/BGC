import lmdb
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from src.dVAE.dVAE import GenomedVAE
from concurrent.futures import ThreadPoolExecutor

PAD_EMBEDDING = torch.ones(320)

class GenomedVAEDataset(Dataset):
    def __init__(self, csv_file, max_workers=64):
        df = pd.read_csv(csv_file)
        self.genome = df['genome'].values
        self.sentence = df['sentence'].values
        self.lmdb_path = df['lmdb path'].values
        self.max_workers = max_workers
        
    def __len__(self):
        return len(self.genome)
    
    def __getitem__(self, idx):
        # unbatched data has single element, batched data has list of elements, using isinstance to handle both cases.
        # genome = self.genome[idx] if isinstance(self.genome[idx], list) else [self.genome[idx]]
        if isinstance(idx, int):
            sentence = self.sentence[idx].split(' ')
            lmdb_path = self.lmdb_path[idx]
            env = lmdb.open(lmdb_path, readonly=True, lock=False)   # lock=False for multi-process reading when using DataLoader
            with env.begin() as txn:
                embed = [pickle.loads(txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence]
                embed = torch.stack(embed)
            
            return embed
            
        sentence = [i.split(' ') for i in self.sentence[idx]]
        lmdb_path = self.lmdb_path[idx]
        
        if self.max_workers:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [executor.submit(self.__get_batch_embed, path, sent) for path, sent in zip(lmdb_path, sentence)]
                embeds = [f.result() for f in futures]
                
            embeds = torch.stack(embeds)
            
            return embeds
        
        embeds = []
        for i, path in enumerate(lmdb_path):
            env = lmdb.open(path, readonly=True, lock=False)
            with env.begin() as txn:
                embed = [pickle.loads(txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence[i]]
                embed = torch.stack(embed)
                embeds.append(embed)
                    
        embeds = torch.stack(embeds)
        
        return embeds
    


    def __get_batch_embed(self, path, sentence):
        env = lmdb.open(path, readonly=True)
        with env.begin() as txn:
            embed = [pickle.loads(txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence]
            embed = torch.stack(embed)
            
        return embed
    
    
    
if __name__ == '__main__':
    train_dataset = GenomedVAEDataset(csv_file='train_100.csv')
    model = GenomedVAE()
    model(train_dataset[0])
    model.tokenize(train_dataset[0:3])
    