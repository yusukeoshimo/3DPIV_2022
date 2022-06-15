import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.my_json import read_json, write_json


if __name__ == '__main__':
    project_dir_path = input('input project dir path >')
    position = input('input position (side or bottom) >')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    control_dict = read_json(json_path)
    target_feature_path = control_dict[position]['target_feature_path']
    model_path = control_dict[position]['LightGBM_model_path']
    input_size = control_dict[position]['features_num'] # 入力のサイズ
    
    
    # モデルの読み込み
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    # 入力と正解ラベルの読み込み
    x = np.memmap(target_feature_path, dtype='float32', mode='r').reshape(-1, input_size)
    y = clf.predict(x)
    
    for value in y:
        if value == 1:
            control_dict = read_json(json_path)
            control_dict[position]['first_pulse'] = int(value)
            write_json(json_path, control_dict)
            break
        if value == 2:
            control_dict = read_json(json_path)
            control_dict[position]['first_pulse'] = int(value)
            write_json(json_path, control_dict)
            break