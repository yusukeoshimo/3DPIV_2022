from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import pickle
import os


def my_classifier(x, y):
    clf = lgbm.LGBMClassifier(objective='multiclass',
                              num_leaves=64,
                              min_child_samples=20,
                              max_depth=4)
    clf.fit(x_train, y_train)
    return clf

if __name__ == '__main__':
    
    x_path = input('input x memmap >')
    y_path = input('input y memmap >')
    
    feature_size = int(input('input size of one data >'))
    save_dir = input('input dir to save LightGBM model >')
    save_path = os.path.join(save_dir, 'my_LightGBM.pkl')
    
    origin_x = np.memmap(x_path, dtype='float32', mode='r').reshape(-1, feature_size)
    origin_y = np.memmap(y_path, dtype='float32', mode='r')
    
    x_train, x_test, y_train, y_test = train_test_split(origin_x, origin_y, test_size=0.2)
    
    clf = my_classifier(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(accuracy)
    print(y_pred)
    
    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    
    with open(save_path, 'rb') as f:
        clf = pickle.load(f)
