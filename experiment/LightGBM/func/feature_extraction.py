import os
from statistics import mode
from cv2 import resize
from lightgbm import cv
import numpy as np
import cv2
from tqdm import tqdm

class ExtractFeatures():
    def __init__(self, arr): # arrには2次元のデータを入れる事．正規化データは使えない
        self.arr = arr.astype('float32')
        self.features = np.array([],dtype='float32')
    
    def extract_diag(self, mag=1): # magはmagnification(倍率)の略
        resize_arr = cv2.resize(self.arr,(int(self.arr.shape[0]*mag), int(self.arr.shape[1]*mag)))
        diag = np.diag(resize_arr)
        self.features = np.hstack((self.features, diag))
    
    def extract_mean(self):
        mean = np.mean(self.arr, dtype='float32')
        self.features = np.hstack((self.features, mean))
    
    def extract_std(self, mag=1):
        resize_arr = cv2.resize(self.arr,(int(self.arr.shape[0]*mag), int(self.arr.shape[1]*mag)))
        std = np.std(resize_arr, dtype='float32')
        self.features = np.hstack((self.features, std))


if __name__ == '__main__':
    memmap_path = input('input memmap path >')
    height = int(input('input height >'))
    width = int(input('input width >'))
    save_dir = input('input dir to save memmap >')
    arr = np.memmap(memmap_path, dtype='uint8', mode='r').reshape(-1, height, width)
    for i, img in enumerate(tqdm(arr)):
        ext = ExtractFeatures(img)
        ext.extract_diag(0.1)
        ext.extract_mean()
        ext.extract_std(0.1)
        if i == 0:
            save_path = os.path.join(save_dir, 'extracted_features_{}.npy'.format(ext.features.shape[0]))
            new_arr = np.memmap(save_path, dtype='float32', mode='w+', shape=(arr.shape[0], ext.features.shape[0]))
        new_arr[i] = ext.features