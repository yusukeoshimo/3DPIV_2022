from operator import imod
from func.video2memmap import video2memmap
from func.mk_label import mk_label
from func.stack_memmap import stack_memmap
from func.feature_extraction import ExtractFeatures
from func.image2memmap import img2memmap
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

if __name__ == '__main__':
    os.chdir(r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM')
    
    # img2memmap
    appending_dir_list = ['appending_{}'.format(i) for i in range(3)]
    appending_memmap_list = ['appending_{}.npy'.format(i) for i in range(3)]
    for i, appending_dir in enumerate(appending_dir_list):
        img2memmap(appending_dir, appending_memmap_list[i])
    
    # make label
    file_num_list = [sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR)) for DIR in appending_dir_list]
    appending_output_list = ['appending_output_{}.npy'.format(i) for i in range(3)]
    for i, file_path in enumerate(appending_dir_list):
        mk_label(file_num_list[i], int(i), appending_output_list[i])
    
    # stack memmap (input)
    appending_raw_data = 'appending_raw_input.npy'
    stack_memmap(appending_memmap_list, appending_raw_data, 960, 1280)
    gc.collect()
    [os.remove(file_name) for file_name in appending_memmap_list]
    
    # stack memmap (output)
    appending_output = 'appending_output.npy'
    stack_memmap(appending_output_list, appending_output, 1, 1)
    gc.collect()
    [os.remove(file_name) for file_name in appending_output_list]
    
    # extract features
    arr = np.memmap(appending_raw_data, dtype='uint8', mode='r').reshape(-1, 960, 1280)
    for i, img in enumerate(tqdm(arr)):
        ext = ExtractFeatures(img)
        ext.extract_std(0.1)
        ext.extract_mean()
        ext.extract_all_values((3, 1))
        if i == 0:
            appending_feature_path = 'appending_features_{}.npy'.format(ext.features.shape[0])
            new_arr = np.memmap(appending_feature_path, dtype='float32', mode='w+', shape=(arr.shape[0], ext.features.shape[0]))
        new_arr[i] = ext.features
    del arr, img
    gc.collect()
    os.remove(appending_raw_data)
    
    # stack features (preleaning+appending)
    prelearning_input = 'prelearning_features_5.npy'
    relearning_input_list = [prelearning_input, appending_feature_path]
    relearning_input = 'relearning_features_5.npy'
    stack_memmap(relearning_input_list, relearning_input, 1, 5)
    del new_arr
    gc.collect()
    [os.remove(file_name) for file_name in relearning_input_list]
    
    # second stack memmap (prelearning_ouput + relearing_output)
    prelearning_output = 'prelearning_label.npy'
    relearning_output_list = [prelearning_output, appending_output]
    relearning_output = 'relearning_output.npy'
    stack_memmap(relearning_output_list, relearning_output, 1, 1)
    gc.collect()
    [os.remove(file_name) for file_name in relearning_output_list]
