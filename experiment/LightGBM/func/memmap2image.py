import os
import numpy as np
import cv2
from tqdm import tqdm

def mmp2img(memmap_path, save_dir, w, h):
    arr = np.memmap(memmap_path, mode='r', dtype='uint8').reshape(-1, h, w)
    for i, frame in enumerate(tqdm(arr)):
        cv2.imwrite(os.path.join(save_dir, '{}.bmp'.format(i)), frame)

if __name__ == '__main__':
    memmap_path = input('input memmap path >')
    save_dir = input('input save dir >')
    w = 1104
    h = 168
    mmp2img(memmap_path, save_dir, w, h)