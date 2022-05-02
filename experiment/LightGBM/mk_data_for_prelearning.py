from func.video2memmap import video2memmap
from func.mk_label import mk_label
from func.stack_memmap import stack_memmap
from func.feature_extraction import ExtractFeatures
import cv2
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.metrics import accuracy_score
import pickle
import time
import gc

def my_classifier(x, y):
    clf = lgbm.LGBMClassifier(objective='multiclass',
                              num_leaves=6,
                              min_child_samples=100,
                              max_depth=2)
    clf.fit(x, y)
    return clf

if __name__ == '__main__':
    os.chdir(r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM')
    
    # video2memmap
    prelearning_video_0 = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\video_for_LightGBM\nr_sc_200.avi'
    prelearning_video_1 = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\video_for_LightGBM\br_bc_200.avi'
    prelearning_video_2 = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\video_for_LightGBM\sr_bc_200.avi'
    prelearning_video_list = [prelearning_video_0, prelearning_video_1, prelearning_video_2]
    prelearning_memmap_list = ['prelearning_{}'.format(i) for i in range(3)]
    for i, prelearning_video in enumerate(prelearning_video_list):
        prelearning_memmap = prelearning_memmap_list[i]
        video2memmap(prelearning_video, prelearning_memmap)
    
    # make label
    prelearning_label_list = ['prelearning_label_{}.npy'.format(i) for i in range(3)]
    for i, video_path in enumerate(prelearning_video_list):
        cap = cv2.VideoCapture(video_path)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        label = i
        prelearning_label = prelearning_label_list[i]
        mk_label(frame_num, label, prelearning_label)
    
    # stack memmap (input)
    stack_prelearning_memap_path = 'prelearning_memmap.npy'
    cap = cv2.VideoCapture(prelearning_video_0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    stack_memmap(prelearning_memmap_list, stack_prelearning_memap_path, height, width)
    for i in prelearning_memmap_list:
        os.remove(i)
    
    # stack memmap (output)
    stack_prelearning_label = 'prelearning_label.npy'
    stack_memmap(prelearning_label_list, stack_prelearning_label, 1, 1)
    for i in prelearning_label_list:
        os.remove(i)
    
    # extract features
    arr = np.memmap(stack_prelearning_memap_path, dtype='uint8', mode='r').reshape(-1, height, width)
    for i, img in enumerate(tqdm(arr)):
        ext = ExtractFeatures(img)
        ext.extract_std(0.1)
        ext.extract_mean()
        ext.extract_all_values((3, 1))
        if i == 0:
            feature_path = r'prelearning_features_{}.npy'.format(ext.features.shape[0])
            new_arr = np.memmap(feature_path, dtype='float32', mode='w+', shape=(arr.shape[0], ext.features.shape[0]))
        new_arr[i] = ext.features
    del arr, img
    gc.collect()
    os.remove(stack_prelearning_memap_path)
    
    # LightGBM
    # x_path = feature_path
    # y_path = stack_prelearning_label
    
    # feature_size = int(ext.features.shape[0])
    # save_dir = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM'
    # save_path = os.path.join(save_dir, 'my_LightGBM.pkl')
    
    # origin_x = np.memmap(x_path, dtype='float32', mode='r').reshape(-1, feature_size)
    # origin_y = np.memmap(y_path, dtype='uint8', mode='r')
    
    # x_train, x_test, y_train, y_test = train_test_split(origin_x, origin_y, test_size=0.1)
    
    # clf = my_classifier(x_train, y_train)
    
    # y_pred = clf.predict(x_test)
    # accuracy = accuracy_score(y_pred, y_test)
    # print(accuracy)
    # print(y_pred)
    
    # with open(save_path, 'wb') as f:
    #     pickle.dump(clf, f)