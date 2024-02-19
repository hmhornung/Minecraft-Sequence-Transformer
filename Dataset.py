from functools import reduce
import torch
import numpy as np
import os
from torch.utils.data import Dataset
# from Minecraft2Token import schemfio
import random
import time

class MinecraftBlockData(Dataset):
    def __init__(self, path: str, files: list[str]):
        self.path = path
        self.files = files
        self.data = []
        for i, file in enumerate(self.files):
            self.data.append(load_ndarray(os.path.join(path, file)))
        print("all data loaded into memory")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index) -> np.ndarray:
        return self.data[index]
    
def custom_collate(batch):
    # batch --> batch_size * X * Y * Z
    batch_size = len(batch)

    # Split the array into 'src' and 'tgt'
    batch = np.stack(batch, axis=0)
    
    src = batch[:,:5]
    tgt = batch[:,5:]
    
    print(src.shape)
    print(tgt.shape)
    
    src = src.reshape((batch_size, -1))
    tgt = tgt.reshape((batch_size, -1))

    sos = np.full((batch_size, 1), fill_value=250) #250 is SOS for decoder in this dataset
    tgt = np.concatenate([sos, tgt], axis=1)

    return {'src': torch.from_numpy(src).to(torch.long), 'tgt': torch.from_numpy(tgt).to(torch.long)}

def get_filenames(path: str, n: int):
    files :list[str] = []
    with os.scandir(path) as entries:
        for i, entry in enumerate(entries):
            if i == 1000000:
                break
            files.append(entry.name)
    print("retrieved filenames")
    return random.sample(files, int(n))

def load_ndarray(path: str) -> np.ndarray:
    with open(path, 'rb') as file:
        all_bytes = file.read()
        
    dtype = np.sctypeDict[np.frombuffer(all_bytes[0:1], np.uint8)[0]]
    ndims = np.frombuffer(all_bytes[1:2], np.uint8)[0]
    arr_start_idx = 2 + (ndims * np.dtype(np.uint32).itemsize)
    dims = np.frombuffer(all_bytes[2:arr_start_idx], np.uint32)

    size = reduce(lambda x, y: x * y, dims) * np.dtype(dtype).itemsize
    arr = np.frombuffer(all_bytes[arr_start_idx: arr_start_idx + size], dtype=dtype)
    arr.resize(dims)
    return arr

def custom_collate_binary(batch):
    # batch --> batch_size * X * Y * Z
    batch_size = len(batch)

    # Split the array into 'src' and 'tgt'
    batch = np.stack(batch, axis=0)
    
    src = batch[:,:5]
    tgt = batch[:,5:]
    
    src[src != 0] = 1
    tgt[tgt != 0] = 1
    
    print(src.shape)
    print(tgt.shape)
    
    src = src.reshape((batch_size, -1))
    tgt = tgt.reshape((batch_size, -1))

    sos = np.full((batch_size, 1), fill_value=2) #2 is SOS for decoder in this dataset
    tgt = np.concatenate([sos, tgt], axis=1)

    return {'src': torch.from_numpy(src).to(torch.long), 'tgt': torch.from_numpy(tgt).to(torch.long)}