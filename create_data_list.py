import lmdb
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp

LMDB_PATH = '/data4/yaoshuai/gtdb_reps_r214/protein_faa_reps/lmdb/'
LOCUS_PATH = '/data4/yaoshuai/gtdb_reps_r214/protein_faa_reps/gene_name/txt'
LMDB_PATH_ALL = '/data5/zhanghaohong/projects/BGC/data/gtdb_all_lmdb/'

def truncate_and_pad(genome_file, max_len=512, stride=8):
    """
    Truncate locus using sliding window. If locus is shorter than max_len, pad it with '<pad>'
    """
    genome = genome_file[:-4]
    
    locus = open(os.path.join(LOCUS_PATH, genome_file)).read().splitlines()
    if isinstance(locus, str):
        locus = [locus] # if genome has only one locus, convert it into 2d list to avoid split gene name into characters.
    locus = [i[1:] if str(i).startswith('>') else i for i in locus] # remove prefix '>'
    
    # truncate locus using sliding window. if locus is shorter than max_len, pad it with '<pad>'
    locus_truncated = []
    
    if len(locus) <= max_len:
        num_patches = 1
        locus_truncated = locus + ['<pad>'] * (max_len - len(locus))
    else:
        for i in range(0, len(locus), stride):
            # truncate locus if it is longer than max_len
            if i + max_len <= len(locus):
                locus_truncated.append(locus[i:i+max_len])
            else:
                truncated = locus[i:]
                truncated += ['<pad>'] * (max_len - len(truncated))
                locus_truncated.append(truncated)
                num_patches = len(locus_truncated)
                break
            
    df =  pd.DataFrame({'genome': [genome] * num_patches, 
                        'sentence': locus_truncated if num_patches > 1 else [locus_truncated],  # if only one patch, convert into 2d list to avoid shape mismatch when creating dataframe.
                        'lmdb path': [os.path.join(LMDB_PATH, genome)] * num_patches})
    return df
    
            
                     
locus_files = os.listdir(LOCUS_PATH)    # 85205 genomes
lmdb_files = os.listdir(LMDB_PATH)  # 85203 genomes

# check inertsection between locus and lmdb
locus_genomes = [i[:-4] for i in locus_files]
intersection = set(locus_genomes).intersection(set(lmdb_files)) # 85203 genomes
locus_files = [i + '.txt' for i in intersection]

# train, val split
train_files, val_files = train_test_split(locus_files, test_size=0.01, random_state=42)

# trancation and padding
# pool = mp.Pool(mp.cpu_count())
# train_df = pool.map(truncate_and_pad, tqdm(train_files))
train_df = [truncate_and_pad(i) for i in tqdm(train_files)]
train_df = pd.concat(train_df).reindex()
train_df['sentence'] = train_df['sentence'].apply(lambda x: ' '.join(x))    # convert list to string
train_df['lmdb path'] = [LMDB_PATH_ALL] * len(train_df) # replace lmdb path with merged lmdb file
train_df.to_csv('train.csv', index=False)
train_df.iloc[:100].to_csv('train_100.csv', index=False)

# val_df = pool.map(truncate_and_pad, tqdm(val_files))
val_df = [truncate_and_pad(i) for i in tqdm(val_files)]
val_df = pd.concat(val_df).reindex()
val_df['sentence'] = val_df['sentence'].apply(lambda x: ' '.join(x))
val_df['lmdb path'] = [LMDB_PATH_ALL] * len(val_df) # replace lmdb path with merged lmdb file
val_df.to_csv('val.csv', index=False)
val_df.iloc[:100].to_csv('val_100.csv', index=False)
