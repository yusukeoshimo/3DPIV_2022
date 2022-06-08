from asyncore import read
from func.video2memmap import video2memmap
from func.feature_extraction import ExtractFeatures
import numpy as np
from tqdm import tqdm
import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..\..'))
from util.my_json import read_json, apend_json, write_json


if __name__ == '__main__':
    project_dir_path = input('input project dir path >')
    position_dir_name = input('input position dir (side or bottom) >')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    control_dict = read_json(json_path)
    cwd = control_dict[position_dir_name]['LightGBM_dir_path']
    os.chdir(cwd)
    
    # video2memmap to improve LightGBM
    target_video = control_dict[position_dir_name]['raw_video_path']
    target_memmap = 'target.npy'
    video2memmap(target_video, target_memmap)
    
    cap = cv2.VideoCapture(target_video) #読み込む動画のパス
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    control_dict = read_json(json_path)
    control_dict[position_dir_name]['video_width'] = width
    control_dict[position_dir_name]['video_height'] = height
    control_dict[position_dir_name]['video_frame_num'] = frame_num
    write_json(json_path, control_dict)
    
    # extract features from improvement video
    arr = np.memmap(target_memmap, dtype='uint8', mode='r').reshape(-1, height, width)
    for i, img in enumerate(tqdm(arr)):
        ext = ExtractFeatures(img)
        ext.extract_std(0.1)
        ext.extract_mean()
        ext.extract_all_values((3, 1))
        if i == 0:
            target_feature = 'target_features_{}.npy'.format(ext.features.shape[0])
            new_arr = np.memmap(target_feature, dtype='float32', mode='w+', shape=(arr.shape[0], ext.features.shape[0]))
        new_arr[i] = ext.features
    
    control_dict = read_json(json_path)
    control_dict[position_dir_name]['features_num'] = ext.features.shape[0]
    write_json(json_path, control_dict)