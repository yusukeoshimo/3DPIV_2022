import os
from statistics import mode
import cv2
import numpy as np
import pandas as pd
import re
import json
from PIL import Image
import glob
from tqdm import tqdm

def mk_label(frame_num, label, memmap_path):
    if frame_num == 0:
        exit('frame_numを0以外にしてください')
    else:
        size = (frame_num)
        memmap = np.memmap(memmap_path, dtype='uint8', mode='w+', shape=size)
        memmap[:] = memmap + label # memmapは［：］がないとファイルの中身が書き変わらない


if __name__ == '__main__':
    label = int(input('input label >'))
    frame_num = int(input('input label num >'))
    file_name = input('input file name >')
    memmap_path = os.path.join(input('input dir to save memmap >'), file_name)
    
    mk_label(frame_num, label, memmap_path)