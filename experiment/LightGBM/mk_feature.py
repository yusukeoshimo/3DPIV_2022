from func.video2memmap import video2memmap
from func.feature_extraction import ExtractFeatures
import numpy as np
from tqdm import tqdm
import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..\..'))
from util.my_json import read_json, apend_json, write_json

class MkFeature():
    def main(self, feature_memmap_path, video_memmap_path, width, height):
        # extract features from improvement video
        arr = np.memmap(video_memmap_path, dtype='uint8', mode='r').reshape(-1, height, width)
        for i, img in enumerate(tqdm(arr)):
            ext = ExtractFeatures(img)
            ext.extract_over_threshold(200)
            ext.extract_over_threshold(150)
            ext.extract_over_threshold(100)
            if i == 0:
                new_arr = np.memmap(feature_memmap_path, dtype='float32', mode='w+', shape=(arr.shape[0], ext.features.shape[0]))
            new_arr[i] = ext.features
    
        self.feature_num = ext.features.shape[0]

if __name__ == '__main__':
    # # video2memmap to improve LightGBM
    video_path = input('input video path >')
    video_memmap_path = input('input videmo memmap path >')
    cap = cv2.VideoCapture(video_path) #読み込む動画のパス
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_dir = input('input save dir >')
    mk_target = MkFeature()
    mk_target.main(os.path.join(save_dir, 'features.npy'), video_memmap_path, width=width, height=height)