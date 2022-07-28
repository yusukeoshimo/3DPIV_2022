import numpy as np
import os
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.my_json import read_json, write_json

def main(save_path,pulse_order,frame_interval, cut_edge ,memmap_dir, height, width, codec):
    if pulse_order == 'second':
        extract_0 = 0+cut_edge # 取得するフレーム．
        extract_1 = extract_0+frame_interval # 取得するフレーム
    elif pulse_order == 'first':
        extract_1 = -cut_edge # 取得するフレーム
        extract_0 = extract_1-frame_interval # 取得するフレーム
        if cut_edge == 0:
            extract_1 = None
    
    file_dict = {int(file_name.split('_')[0]) : file_name for file_name in os.listdir(memmap_dir)} # {xxx:'xxx_yyy.npy', ....}
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    fps = 1 # 作成する動画のfps
    video = cv2.VideoWriter(save_path,fourcc,fps,(width, height))
    for i in tqdm(range(len(file_dict))):
        arr = np.memmap(os.path.join(memmap_dir, file_dict[i]), dtype='uint8', mode='r').reshape(-1, height, width)
        for frame in [arr[extract_0], arr[extract_1]]:
            frame = np.repeat(frame, 3).reshape(height,width,3).astype(np.uint8)
            video.write(frame)
    video.release()

if __name__ == '__main__':
    memmap_dir = input('input dir path saved memmap classified by pulse state >') # memmapが保存されているディレクトリ．order_fps.npyで保存すること
    codec = 'Y800'
    save_dir = input('input save dir >') # 作成した動画を保存するディレクトリ
    frame_interval = 30 # 取得フレームと取得フレームの差．1~
    cut_edge = 1 # 動画の両端をいくら切り捨てるか．0~
    save_path = os.path.join(save_dir, 'FrameInterval_{}.avi'.format(frame_interval))
    height = 168 # フレームの高さ
    width = 1104 # フレームの幅
    pulse_order = 'second' # 最初に点灯するレーザー．1 or 2
    main(save_path,pulse_order,frame_interval, cut_edge ,memmap_dir, height, width, codec)