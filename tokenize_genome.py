import pandas as pd
import torch
from src.dVAE.dVAE import GenomedVAE
from src.dVAE.Dataset import GenomedVAEDataset
from tqdm import tqdm
from pickle import dump

class GenomeTokenizer:
    def __init__(self, model, sliding_window=512, stride=256, map_window=10):
        self.model = model
        self.sliding_window = sliding_window
        self.stride = stride
        self.map_window = map_window

    def map_tokens(self, recent, map, is_pre_map):
        if is_pre_map:
            recent[:self.map_window] = map[self.stride:self.stride+self.map_window] # map the first map_window tokens with the previous sentence
        else:
            recent[-self.map_window:] = map[self.sliding_window-self.stride-self.map_window:self.sliding_window-self.stride] # map the last map_window tokens with the next sentence
        return recent

    def tokenize_and_mapping(self, data):
        # tokenize
        print('tokenizing...')
        tokens = torch.stack([self.model.tokenize(**sent)[0] for sent in tqdm(data)])
        
        # mapping
        print('mapping...')
        for i, (genome, token) in enumerate(zip(data.genome, tokens)):
            pre_genome = data.genome[i - 1] if i > 0 else None
            post_genome = data.genome[i + 1] if i < len(data) - 1 else None
            
            if (i == 0 or i == len(data) - 1) and (pre_genome != genome) and (post_genome != genome):
                continue
            
            if genome == pre_genome:
                tokens[i] = self.map_tokens(token, tokens[i-1], is_pre_map=True)
                
            if genome == post_genome:
                tokens[i] = self.map_tokens(token, tokens[i+1], is_pre_map=False)
                
        return tokens

if __name__ == '__main__':
    val_set = GenomedVAEDataset(csv_file='val_100.csv')
    model = GenomedVAE.load_from_checkpoint('lightning_logs/dVAE_2048/version_3/checkpoints/epoch=0-step=24650.ckpt', map_location=torch.device('cpu'))

    # toenize
    # model.eval()
    model.freeze()

    tokenizer = GenomeTokenizer(model)
    tokens = tokenizer.tokenize_and_mapping(val_set)
    # convert token to str with '' separator
    tokens = [' '.join([str(t.item()) for t in token]) for token in tokens]
    tokens = [[sent, token] for sent, token in zip(val_set.sentence, tokens)]
    
    print('saving...')
    pd.DataFrame(tokens, columns=['sentence', 'token']).to_csv('data/val_tokens.csv', index=False)
    print('val tokens done')
    
    del val_set
    del tokens
    
    train_set = GenomedVAEDataset(csv_file='train.csv')
    tokens = tokenizer.tokenize_and_mapping(train_set)
    tokens = {sent: token for sent, token in zip(train_set.sentence, tokens)}
    
    print('saving...')
    pd.DataFrame(tokens, columns=['sentence', 'token']).to_csv('data/train_tokens.csv', index=False)
    print('train tokens done')
    