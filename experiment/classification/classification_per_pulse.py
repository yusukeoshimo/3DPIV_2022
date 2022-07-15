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


def all_remove(arr, value):
    while value in arr:
        arr.remove(value)

def index_block(arr, value):
    # arr: 1次元配列
    # value: 取得したい値．

    # [[value, ..., value], [], [], ..., [value, ..., value]]
    classificate_list = []
    list_2 = []
    for i, label in enumerate(arr):
        if label != value:
            classificate_list.append(list_2)
            list_2 = []
        else:
            list_2.append(i)
    classificate_list.append(list_2)
    
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
    fpp = int(fps*spp) # frame per pulse
    fpi = int(fps*itr_time) # frame per one iteration
    fpc = int(fps*cooldown_time) # frame per cool_time
    
    
    # モデルの読み込み
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    # 入力と正解ラベルの読み込み
    x = np.memmap(target_feature_path, dtype='float32', mode='r').reshape(-1, input_size)
    y = clf.predict(x)
    
    # 動画をmemmapに変換したデータを読み込み
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, height, width)
    
    index_list = index_block(y, 0) # 0の状態のフレームのインデックスをまとめたリストを作成
    index_list = [i for i in index_list if len(i) >= 0.1*fpc] # first pulse と second pulse の間の0状態のフレームを取り除く
    del index_list[-1] # 右端の0の状態のindexを削除する
    del index_list[-1] # 右端の0の状態のindexを削除する
    
    # 誤分類の確認
    for i, index in enumerate(index_list):
        if i != len(index_list)-1:
            next_index = index_list[i+1]
            if abs(2*fpp-((next_index[0]-1)-(index[-1]+1)+1))/(2*fpp) >= 0.1:
                exit('誤分類の可能性あり：{}~{}'.format(index[-1], next_index[0]))
    
    # 作成したリストのフレームのみをmemmapに書き込む
    for i, index in enumerate(tqdm(index_list)):
        start_point = index[-1] + 1
        end_point = start_point+(2*fpp)
        pulse_frames = video_mem[start_point:end_point]
        if position == 'side':
            data = pulse_frames[-fpp:]
        if position == 'bottom':
            data = pulse_frames[:fpp]
        size = data.shape
        file_name = '{}_{}.npy'.format(i, size[0])
        arr = np.memmap(file_name, dtype='uint8', mode='w+', shape=size)
        arr[:] = data