from turtle import position
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..\..'))
from util.my_json import read_json, apend_json, write_json



if __name__ == '__main__':
    project_dir_path = input('input project dir >')
    position = input('input position (side or bottom) >')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    control_dict = read_json(json_path)
    cwd = control_dict[position]['LightGBM_dir_path']
    os.chdir(cwd)
    memmap_path = control_dict[position]['target_feature_path']
    model_path = control_dict[position]['LightGBM_model_path']
    data_len = control_dict[position]['features_num']
    dir_0 = control_dict[position]['0_dir_path']
    dir_1 = control_dict[position]['1_dir_path']
    dir_2 = control_dict[position]['2_dir_path']
    width = control_dict[position]['video_width']
    height = control_dict[position]['video_height']
    video_mem_path = control_dict[position]['target_memmap_path']
    
    # ディレクトリ内を空にする
    dir_list = [dir_0, dir_1, dir_2]
    for i in dir_list:
        if os.path.exists(i):
            shutil.rmtree(i)
        os.mkdir(i)
    
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    x = np.memmap(memmap_path, dtype='float32', mode='r').reshape(-1, data_len)
    y = clf.predict(x)
    
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, height, width)
    
    for i, img in enumerate(tqdm(video_mem)):
        if y[i] == 0:
            cv2.imwrite(os.path.join(dir_0,'{}.bmp'.format(i)), img)
        elif y[i] == 1:
            cv2.imwrite(os.path.join(dir_1,'{}.bmp'.format(i)), img)
        elif y[i] == 2:
            cv2.imwrite(os.path.join(dir_2,'{}.bmp'.format(i)), img)