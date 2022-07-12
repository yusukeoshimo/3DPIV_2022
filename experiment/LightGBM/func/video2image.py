import os
import cv2
import numpy as np
import pandas as pd
import re
import json
from PIL import Image
import glob
from tqdm import tqdm

def video_processing(video_path, save_dir):
    def wrapper(*args, **kwargs):
        cap = cv2.VideoCapture(video_path) #読み込む動画のパス
        
        output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            
            cv2.imwrite(os.path.join(save_dir, '{}.bmp'.format(i)), frame)
            
        cap.release()
    return wrapper

if __name__ == '__main__':
    original_video_path = input('original video path >')
    save_dir = input('input save dir >')
    video_processing(original_video_path, save_dir)()
