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

def video2label(original_video_path, label, memmap_path):
    video_path = original_video_path
    cap = cv2.VideoCapture(video_path) #読み込む動画のパス
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (frame_num)
    cap.release()
    
    memmap = np.memmap(memmap_path, dtype='uint8', mode='w+', shape=size)
    memmap[:] = memmap + label # memmapは［：］がないとファイルの中身が書き変わらない


if __name__ == '__main__':
    original_video_path = input('original video path >')
    label = int(input('input label >'))
    file_name = os.path.splitext(os.path.basename(original_video_path))[0]
    memmap_path = os.path.join(input('input dir to save memmap >'), '{}_output_{}.npy'.format(file_name, label))
    
    video2label(original_video_path, label, memmap_path)