import numpy as np
import pickle
from tqdm import tqdm
import os
import numpy as np
from tqdm import tqdm
import math
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.my_json import read_json, write_json
from util.txt_replacement import extract_txt


def index_block(arr, value):
    # arr: 1次元配列
    # value: 取得したい値．
    # [[value, ..., value], [], [], ..., [value, ..., value]]
    classificate_list = []
    list_2 = []
    for i, label in enumerate(arr):
        if label != value:
            list_2 = []
            classificate_list.append(list_2)
        else:
            list_2.append(i)
    
    return [i for i in classificate_list if i != []] # [[value, ..., value], ..., [value, ..., value]]


if __name__ == '__main__':
    project_dir_path = input('input project dir path >')
    position = input('input position (side or bottom) >')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    control_dict = read_json(json_path)
    cwd = control_dict[position]['data_2_memmap_dir_path'] # ここにバグありそう
    os.chdir(cwd)
    target_feature_path = control_dict[position]['target_feature_path']
    model_path = control_dict[position]['LightGBM_model_path']
    input_size = control_dict[position]['features_num'] # 入力のサイズ
    width = control_dict[position]['video_width'] # フレームの幅
    height = control_dict[position]['video_height'] # フレームの高さ
    fps = control_dict[position]['video_fps'] # fps
    video_mem_path = control_dict[position]['target_memmap_path']
    
    
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'laser_blinking', 'laser_blinking.ino'), 'r') as f:
        arduino_src = f.read()
    cooldown_time = int(extract_txt(arduino_src, 'int cooldown_time = ', ';')[0])/1000
    spp = int(extract_txt(arduino_src, 'int turn_on_time = ', ';')[0])/1000 # second per pulse
    itr_time = cooldown_time + 2*spp
    fpp = fps*spp # frame per pulse
    fpi = fps*itr_time # frame per one iteration
    
    
    # モデルの読み込み
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    # 入力と正解ラベルの読み込み
    x = np.memmap(target_feature_path, dtype='float32', mode='r').reshape(-1, input_size)
    y = clf.predict(x)
    
    # 動画をmemmapに変換したデータを読み込み
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, height, width)
    
    index_lists_0 = index_block(y, 0) # 2の状態のフレームのインデックスをまとめたリストを作成
    del index_lists_0[0]
    del index_lists_0[-1]
    index_lists_1 = index_block(y, 1) # 2の状態のフレームのインデックスをまとめたリストを作成
    index_lists_2 = index_block(y, 2) # 2の状態のフレームのインデックスをまとめたリストを作成
    
    misclassification_list = []
    
    print(index_lists_0[0])
    
    for i, index_list_0 in enumerate(tqdm(index_lists_0)):
        if abs((len(index_list_0)-cooldown_time*fps)/(cooldown_time*fps)) >= 0.1:
            misclassification_list.append(index_list_0)
    
    for i, index_list_1 in enumerate(tqdm(index_lists_1)):
        if abs((len(index_list_1)-spp*fps)/(spp*fps)) >= 0.1:
            misclassification_list.append(index_list_1)
    
    for i, index_list_2 in enumerate(tqdm(index_lists_2)):
        if abs((len(index_list_2)-spp*fps)/(spp*fps)) >= 0.1:
            misclassification_list.append(index_list_2)
    
    print('誤判定の疑いがあるフレーム : {}'.format(misclassification_list))