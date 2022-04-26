import os
import numpy as np
import cv2
from tqdm import tqdm

def img2memmap(dir_path, memmap_path):
    paths = os.listdir(dir_path) # ディレクトリ内のファイルパスを全てリストに格納
    for i, path in enumerate(tqdm(paths)):
        img = np.array(cv2.imread(os.path.join(dir_path, path), 0))
        
        # memmapの準備
        if i == 0:
            memmap = np.memmap(memmap_path, dtype='uint8', mode='w+', shape=(len(paths), img.shape[0], img.shape[1]))
        
        memmap[i] = img # memmapの書き込み

if __name__ == '__main__':
    dir_path = input('input dir saved images >')
    memmap_path = os.path.join(input('input dir to save memmap >'), 'im2mem.npy')
    
    img2memmap(dir_path, memmap_path)