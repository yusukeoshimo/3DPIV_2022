import os
import sys
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

src_memmap_path = input('input src memmap > ')
src_memmap = np.memmap(src_memmap_path, mode='r', dtype=np.float16).reshape(-1, 3)

dst_memmap_path = os.path.join(input('input save dir >'), 'abs.npy')
dst_memmap = np.memmap(dst_memmap_path, mode='w+', dtype=src_memmap.dtype, shape=src_memmap.shape)

for i, data in enumerate(tqdm(src_memmap)):
    dst_memmap[i] = np.hstack((data[0], data[1], np.abs(data[2])))