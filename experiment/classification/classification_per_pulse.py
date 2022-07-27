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
            classificate_list.append(list_2)
            list_2 = []
        else:
            list_2.append(i)
    classificate_list.append(list_2)
    
    return [i for i in classificate_list if i != []] # [[value, ..., value], ..., [value, ..., value]]

def main(save_dir, model_path, target_feature_path, input_size, video_mem_path, width, height, fps, pulse_order, cooldown_time, turn_on_time):
    fpp = int(fps*turn_on_time) # frame per pulse
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
        if pulse_order == 'first':
            data = pulse_frames[:fpp]
        if pulse_order == 'second':
            data = pulse_frames[-fpp:]
        size = data.shape
        save_path = os.path.join(save_dir, '{}_{}.npy'.format(i, size[0]))
        arr = np.memmap(save_path, dtype='uint8', mode='w+', shape=size)
        arr[:] = data

if __name__ == '__main__':
    save_dir = input('input save dir >')
    model_path = input('input LightGBM model path >')
    target_feature_path = input('input feature path >')
    input_size = 3 # 入力のサイズ
    video_mem_path = input('input video memmap path >')
    width = 1104 # フレームの幅
    height = 168 # フレームの高さ
    fps = 120 # fps
    pulse_order = 'second'
    cooldown_time = 0.1
    turn_on_time = 0.5
    
    main(save_dir, model_path, target_feature_path, input_size, video_mem_path, width, height, fps, pulse_order, cooldown_time, turn_on_time)