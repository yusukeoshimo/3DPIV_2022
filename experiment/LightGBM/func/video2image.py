import os
import cv2
import numpy as np
import pandas as pd
import re
import json
from PIL import Image
import glob
from tqdm import tqdm

def video_processing(original_video_path, calibed_video_path, f):
    print(2)
    def wrapper(*args, **kwargs):
        video_path = original_video_path
        cap = cv2.VideoCapture(video_path) #読み込む動画のパス
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # コーデックの取得
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = list((fourcc_int.to_bytes(4,'little').decode('utf-8')))
        print('fourcc', fourcc)
        
        
        output_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(fourcc[0],fourcc[1],fourcc[2], fourcc[3]) #mp4フォーマット
        # fourcc = cv2.VideoWriter_fourcc('Y','8','0', '0') #mp4フォーマット
        # fourcc = cv2.VideoWriter_fourcc('D','I','B', '') #mp4フォーマット
        video = cv2.VideoWriter(calibed_video_path, fourcc, fps, output_size) #書き込み先のパス、フォーマット、fps、サイズ(幅×高さ)

        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            
            # ここに処理を記入********************************************
            frame = np.zeros((output_size[1],output_size[0],3), np.uint8)+frame
            # **********************************************************
            video.write(frame)
            cv2.imwrite(r'C:\Users\yusuk\Desktop\m2\orthogonal\{}.bmp'.format(i), frame)
            
        cap.release()
        video.release()
    return wrapper

if __name__ == '__main__':
    original_video_path = input('original video path >')
    calibed_video_path = os.path.join(input('dir path saved calibed-video >'), 'calibed_video.avi')
    f = 1
    video_processing(original_video_path, calibed_video_path, f)()
