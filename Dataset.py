from functools import reduce
import torch
import numpy as np
import os
from torch.utils.data import Dataset
# from Minecraft2Token import schemfio
import time

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

class MinecraftBlockData(Dataset):
    def __init__(self, path: str, files: list[str]):
        self.path = path
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index) -> np.ndarray:
        return load_ndarray(os.path.join(self.path, self.files[index]))
    
def custom_collate(batch):
    # batch --> batch_size * X * Y * Z
    batch_size = batch.shape[0]

    # Split the array into 'src' and 'tgt'
    src = batch[:, :5]
    tgt = batch[:, 5:]

    src = src.reshape((batch_size, -1))
    tgt = tgt.reshape((batch_size, -1))

    sos = np.full((batch_size, 1), fill_value=255, dtype=np.uint8) #255 is gonna be the start of sequence token for decoder
    tgt = np.concatenate([sos, tgt], axis=1)

    return {'src': torch.from_numpy(src), 'tgt': torch.from_numpy(tgt)}

