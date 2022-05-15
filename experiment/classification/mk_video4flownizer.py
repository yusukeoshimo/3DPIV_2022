import numpy as np
import os
import cv2
import time
from tqdm import tqdm


if __name__ == '__main__':
    dir_path = input('input dir path saved memmap >')
    os.chdir(dir_path)
    save_dir = input('input save dir >')
    step = 2
    cut_edge = 2 # 動画の両端をいくら切り捨てるか
    pulse_order = 'second'
    if pulse_order == 'second':
        extract_0 = 0+cut_edge
        extract_1 = extract_0+step
    else:
        extract_1 = -1-cut_edge
        extract_0 = -extract_1-step
    save_path = os.path.join(save_dir, 'step_{}.avi'.format(step))
    height = 960
    width = 1280
    fps = 1
    
    
    file_dict = {int(file_name.split('_')[0]) : file_name for file_name in os.listdir(dir_path)} # {xxx:'xxx_yyy.npy', ....}
    
    fourcc = cv2.VideoWriter_fourcc('Y','8','0','0')
    video = cv2.VideoWriter(save_path,fourcc,fps,(width, height))
    for i in tqdm(range(len(file_dict))):
        arr = np.memmap(file_dict[i], dtype='uint8', mode='r').reshape(-1, height, width)
        for frame in [arr[extract_0], arr[extract_1]]:
            frame = np.repeat(frame, 3).reshape(height,width,3).astype(np.uint8)
            video.write(frame)
    video.release()