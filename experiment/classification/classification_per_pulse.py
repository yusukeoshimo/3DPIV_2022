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
            list_2 = []
            classificate_list.append(list_2)
        else:
            list_2.append(i)
    
    return [i for i in classificate_list if i != []] # [[value, ..., value], ..., [value, ..., value]]

def improve_index_list(index_list, fpp, fpi):
    # index_list: [[idx1, ..., idxk], ..., [idxl, ..., idxm]]
    # fpp : frame per one pulse
    # fpi : frame per iteration
    fpp = math.ceil(fpp)
    improved_list = []
    for i, indexes in enumerate(index_list):
        if i == 0:
            if abs((len(indexes)-fpp)/fpp) <= 0.1:
                start_point = indexes[0]
                improved_list.append(list(range(start_point, start_point+fpp)))
            else:
                exit('!!! my error 1 !!!')
        else:
            print(abs((indexes[0]-(start_point+fpi))/fpi))
            if abs((indexes[0]-(start_point+fpi))/fpi) <= 0.1:
                start_point = indexes[0]
                improved_list.append(list(range(start_point, start_point+fpp)))
    return improved_list

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
    
    index_list = index_block(y, 2) # 2の状態のフレームのインデックスをまとめたリストを作成
    index_list = improve_index_list(index_list, fpp, fpi) # 作成したリストの誤判定を丸める
    
    # 作成したリストのフレームのみをmemmapに書き込む
    for i, index in enumerate(tqdm(index_list)):
        data = video_mem[index]
        size = data.shape
        file_name = '{}_{}.npy'.format(i, size[0])
        arr = np.memmap(file_name, dtype='uint8', mode='w+', shape=size)
        arr[:] = data
    
    # 誤判定が無いか確認する
    fpp = fps*spp
    print('誤分類が無いか確認中...')
    for i, indexes in enumerate(index_list):
        if len(indexes) <= fpp*0.9:
            print('誤分類：{}'.format(i))
            print('LightGBMをトレーニングし直してください')