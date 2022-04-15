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

def video2memmap(original_video_path, memmap_path):
    video_path = original_video_path
    cap = cv2.VideoCapture(video_path) #読み込む動画のパス
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_num, height, width)
    
    memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=size)
    print(memmap.shape)

    for i in tqdm(range(frame_num)):
        ret, frame = cap.read()
        frame = frame[:,:,0] #(height, width, 3) -> (height, width)

        memmap[i] = frame # 書き込み
        
    cap.release()

if __name__ == '__main__':
    original_video_path = input('original video path >')
    file_name = os.path.splitext(os.path.basename(original_video_path))[0]
    memmap_path = os.path.join(input('input dir to save memmap >'), '{}_input.npy'.format(file_name))

    video2memmap(original_video_path, memmap_path)