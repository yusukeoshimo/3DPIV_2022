import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

def extract_background(save_path, memmap_dir, w, h, left_cut, right_cut):
    files = os.listdir(memmap_dir)
    paths = [os.path.join(memmap_dir, file_name) for file_name in files]
    
    background_list = []
    for file_path in paths:
        arr = np.memmap(file_path, dtype='uint8', mode='r').reshape(-1, h, w)[left_cut:-right_cut] # レーザーが点灯しきっていないエッジをカットする
        background = np.amin(arr, axis=0)
        background_list.append(background)
    background = np.amin(np.array(background_list), axis=0)
    return background

def subtract(save_dir, memmap_dir, background_img):
    files = os.listdir(memmap_dir)
    paths = [os.path.join(memmap_dir, file_name) for file_name in files]
    for memmap_path in paths:
        arr = np.memmap(memmap_path, mode='r', dtype=np.uint8).reshape(-1, background_img.shape[0], background_img.shape[1])
        background_arr = background_img.reshape(-1, background_img.shape[0], background_img.shape[1])
        background_arr = np.repeat(background_arr, arr.shape[0], axis=0)
        subtract_arr = cv2.subtract(arr, background_arr)
        subtract_memmap = np.memmap(os.path.join(save_dir, os.path.basename(memmap_path)), dtype=np.uint8, mode='w+', shape=arr.shape)
        subtract_memmap[:] = subtract_arr[:]

if __name__ == '__main__':
    memmap_dir = input('input dir path saved memmap >')
    
    w = 1104
    h = 168
    left_cut = 2
    right_cut = 2
    
    background_img = extract_background(save_path, memmap_dir, w, h, left_cut, right_cut)
    
    save_dir = input('input dir to save memmap >')
    memmap_dir = input('input memmap_dir >')
    subtract(save_dir, memmap_dir, background_img)