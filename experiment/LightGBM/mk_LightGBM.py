from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import pickle
import os


def my_classifier(x, y):
    clf = lgbm.LGBMClassifier(objective='multiclass',
                              num_leaves=40,
                              min_child_samples=100,
                              max_depth=2)
    clf.fit(x, y)
    return clf

if __name__ == '__main__':
    os.chdir(input('input cwd >'))
    
    x_path = input('input x memmap >')
    y_path = input('input y memmap >')
    
    feature_size = int(input('input size of one data >'))
    save_path = 'my_LightGBM.pkl'
    
    origin_x = np.memmap(x_path, dtype='float32', mode='r').reshape(-1, feature_size)
    origin_y = np.memmap(y_path, dtype='uint8', mode='r')
    
    
    clf = my_classifier(origin_x, origin_y)
    
    
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    
    with open(save_path, 'rb') as f:
        clf = pickle.load(f)
