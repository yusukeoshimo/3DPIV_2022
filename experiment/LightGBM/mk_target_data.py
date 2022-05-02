from func.video2memmap import video2memmap
from func.feature_extraction import ExtractFeatures
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    os.chdir(r'C:\Users\yusuk\Desktop\3DPIV_2022\data\memmap_for_LightGBM')
    
    # video2memmap to improve LightGBM
    target_video = r'C:\Users\yusuk\Desktop\3DPIV_2022\data\test_video_for_LightGBM\横カメラ_先取得.avi'
    target_memmap = 'target.npy'
    video2memmap(target_video, target_memmap)
    
    # extract features from improvement video
    height = 960
    width = 1280
    arr = np.memmap(target_memmap, dtype='uint8', mode='r').reshape(-1, height, width)
    for i, img in enumerate(tqdm(arr)):
        ext = ExtractFeatures(img)
        ext.extract_std(0.1)
        ext.extract_mean()
        ext.extract_all_values((3, 1))
        if i == 0:
            target_feature = 'target_features_{}.npy'.format(ext.features.shape[0])
            new_arr = np.memmap(target_feature, dtype='float32', mode='w+', shape=(arr.shape[0], ext.features.shape[0]))
        new_arr[i] = ext.features