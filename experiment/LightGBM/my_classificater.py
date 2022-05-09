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
    dir_2 = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM\2'
    dir_1 = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM\1'
    dir_0 = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM\0'
    width = 960
    height = 1280
    video_mem_path = input('input target memmap >')
    
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    x = np.memmap(memmap_path, dtype='float32', mode='r').reshape(-1, data_len)
    y = clf.predict(x)
    
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, width, height)
    
    for i, img in enumerate(tqdm(video_mem)):
        if y[i] == 0:
            cv2.imwrite(os.path.join(dir_0,'{}.bmp'.format(i)), img)
        elif y[i] == 1:
            cv2.imwrite(os.path.join(dir_1,'{}.bmp'.format(i)), img)
        elif y[i] == 2:
            cv2.imwrite(os.path.join(dir_2,'{}.bmp'.format(i)), img)