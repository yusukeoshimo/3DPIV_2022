import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

def avg_memmaps(dir_path, w, h, left_cut, right_cut):
    files = os.listdir(dir_path)
    paths = [os.path.join(dir_path, file_name) for file_name in files]
    
    intensity_avg_list = []
    for file_path in paths:
        arr = np.memmap(file_path, dtype='uint8', mode='r').reshape(-1, h, w)[left_cut:-right_cut] # レーザーが点灯しきっていないエッジをカットする
        intensity_avg = np.mean(arr, axis=0)
        intensity_avg_list.append(intensity_avg)
    intensity_avg = np.mean(np.array(intensity_avg_list), axis=0)
    return intensity_avg

if __name__ == '__main__':
    dir_path = input('input dir path saved memmap >')
    save_dir = input('input save dir >')
    
    w = 1104
    h = 168
    
    left_cut = 2
    right_cut = 2
    
    intensity_avg = avg_memmaps(dir_path, w, h, left_cut, right_cut)
    save_path = os.path.join(save_dir, 'intensity_avg.bmp')
    cv2.imwrite(save_path, intensity_avg)