from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..\..'))
from util.my_json import read_json, apend_json, write_json


def my_classifier(x, y):
    clf = lgbm.LGBMClassifier(objective='binary',
                              num_leaves=40,
                              min_child_samples=100,
                              max_depth=2)
    # clf = lgbm.LGBMClassifier(objective='multiclass',
    #                           num_leaves=40,
    #                           min_child_samples=100,
    #                           max_depth=2)
    clf.fit(x, y)
    return clf

if __name__ == '__main__':
    project_dir_path = input('input project dir path >')
    position_dir_name = input('input position dir (side or bottom) >')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    control_dict = read_json(json_path)
    cwd = control_dict[position_dir_name]['LightGBM_dir_path']
    os.chdir(cwd)
    
    x_path = control_dict[position_dir_name]['learning_input_path']
    y_path = control_dict[position_dir_name]['learning_label_path']
    
    feature_size = control_dict[position_dir_name]['features_num']
    save_path = control_dict[position_dir_name]['LightGBM_model_path']
    
    origin_x = np.memmap(x_path, dtype='float32', mode='r').reshape(-1, feature_size)
    origin_y = np.memmap(y_path, dtype='uint8', mode='r')
    
    
    clf = my_classifier(origin_x, origin_y)
    
    
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    
    with open(save_path, 'rb') as f:
        clf = pickle.load(f)
