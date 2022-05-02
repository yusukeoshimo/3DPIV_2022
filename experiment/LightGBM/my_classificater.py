from statistics import mode
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os

if __name__ == '__main__':
    memmap_path = input('input x >')
    model_path = input('input model >')
    data_len = 5
    posi_path = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM\2'
    nega_path = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM\1'
    black_path = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM\0'
    video_mem_path = input('input target memmap >')
    
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    x = np.memmap(memmap_path, dtype='float32', mode='r').reshape(-1, data_len)
    y = clf.predict(x)
    
    posi_idx = y == 2
    nega_idx = y == 1
    black_idx = y == 0
    
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, 960, 1280)
    
    for i, img in enumerate(tqdm(video_mem[posi_idx])):
        cv2.imwrite(os.path.join(posi_path,'_{}.bmp'.format(i)), img)

    for i, img in enumerate(tqdm(video_mem[nega_idx])):
        cv2.imwrite(os.path.join(nega_path,'_{}.bmp'.format(i)), img)

    for i, img in enumerate(tqdm(video_mem[black_idx])):
        cv2.imwrite(os.path.join(black_path,'_{}.bmp'.format(i)), img)