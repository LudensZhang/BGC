import lmdb
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from src.dVAE.dVAE import GenomedVAE
from concurrent.futures import ThreadPoolExecutor

PAD_EMBEDDING = torch.zeros(320)

class GenomedVAEDataset(Dataset):
    def __init__(self, csv_file, lmdb_in_one=True, max_workers=64):
        df = pd.read_csv(csv_file)
        self.genome = df['genome'].values
        self.sentence = df['sentence'].values
        self.lmdb_path = df['lmdb path'].values
        self.lmdb_in_one = lmdb_in_one
        if not lmdb_in_one:
            print('lmdb_in_one is False, this will cause slow reading speed. We recommend to merge all lmdb files into one for faster reading.')
        else:
            self.env = lmdb.open(self.lmdb_path[0], readonly=True)
            self.txn = self.env.begin()
        self.max_workers = max_workers
        
    def __len__(self):
        return len(self.genome)
    
    def __getitem__(self, idx):
        # unbatched data has single element, batched data has list of elements, using isinstance to handle both cases.
        # genome = self.genome[idx] if isinstance(self.genome[idx], list) else [self.genome[idx]]
        
        if not self.lmdb_in_one:    # if lmdb files are not merged into one, open lmdb file for sentence one by one.
            return self.__getitem__legacy(idx)
        
        if isinstance(idx, int):
            sentence = np.array(self.sentence[idx].split(' '))
            mask = torch.tensor(sentence != '<pad>').float()
            embed = [pickle.loads(self.txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence]
            embed = torch.stack(embed)
            return {'x': embed, 'mask': mask}
        
        sentence = np.array([i.split(' ') for i in self.sentence[idx]])
        
        embeds = []
        masks = []
        
        masks = torch.tensor(sentence != '<pad>').float()
        
        for sent in sentence:
            mask = torch.tensor(sent != '<pad>').float()
            
            embed = [pickle.loads(self.txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sent]
            embed = torch.stack(embed)
            embeds.append(embed)
            
        embeds = torch.stack(embeds)
        
        return {'x': embeds, 'mask': masks}          
    
    def __getitem__legacy(self, idx):
        if isinstance(idx, int):
            sentence = np.array(self.sentence[idx].split(' '))
            mask = torch.tensor(sentence != '<pad>').float()
            lmdb_path = self.lmdb_path[idx]
            env = lmdb.open(lmdb_path, readonly=True, lock=False)   # lock=False for multi-process reading when using DataLoader
            with env.begin() as txn:
                embed = [pickle.loads(txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence]
                embed = torch.stack(embed)
            
            return {'x': embed, 'mask': mask}
            
        sentence = [np.array(i.split(' ')) for i in self.sentence[idx]]
        lmdb_path = self.lmdb_path[idx]
        
        if self.max_workers:
            with ThreadPoolExecutor(self.max_workers) as executor:
                futures = [executor.submit(self.__get_batch_embed_legacy, path, sent) for path, sent in zip(lmdb_path, sentence)]
                embeds = [f.result() for f in futures]
                
            masks = [torch.tensor(sent != '<pad>').float() for sent in sentence]
                
            embeds = torch.stack(embeds)
            masks = torch.stack(masks)
            
            return {'x': embeds, 'mask': masks}
        
        embeds = []
        masks = []
        
        for i, path in enumerate(lmdb_path):
            env = lmdb.open(path, readonly=True, lock=False)
            with env.begin() as txn:
                mask = torch.tensor(sentence[i] != '<pad>').float()
                embed = [pickle.loads(txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence[i]]
                embed = torch.stack(embed)
                embeds.append(embed)
                masks.append(mask)
                    
        embeds = torch.stack(embeds)
        
        return embeds

    def __get_batch_embed_legacy(self, path, sentence):
        env = lmdb.open(path, readonly=True, lock=False)
        with env.begin() as txn:
            embed = [pickle.loads(txn.get(str(word).encode('ascii')))['mean_representations'][6] if word != '<pad>' else PAD_EMBEDDING for word in sentence]
            embed = torch.stack(embed)
            
        return embed
    
    
    
if __name__ == '__main__':
    train_dataset = GenomedVAEDataset(csv_file='train_100.csv')
    model = GenomedVAE()
    model(**train_dataset[0])
    model.tokenize(**train_dataset[0:3])
    
    # read time test
    import time
    start = time.time()
    tmp = train_dataset[:]
    end = time.time()
    print(f'LMDB in one file, time: {end-start}')
    
    start = time.time()
    tmp = [train_dataset[i] for i in range(len(train_dataset))]
    end = time.time()
    print(f'No batch, time: {end-start}')
    
    train_dataset.lmdb_in_one = False
    
    for workers in [None, 16, 32, 64]:
        train_dataset.max_workers = workers
        start = time.time()
        tmp = train_dataset[:]
        end = time.time()
        print(f'workers: {workers}, time: {end-start}')