import cv2
import numpy as np
import pickle
from tqdm import tqdm
import os
import shutil
import numpy as np
from tqdm import tqdm

def all_remove(arr, value):
    while value in arr:
        arr.remove(value)

def index_block(arr, value):
    classificate_list = []
    for i, label in enumerate(y):
        if label != value:
            list_2 = []
            classificate_list.append(list_2)
        else:
            list_2.append(i)
    return [i for i in classificate_list if i != []]

if __name__ == '__main__':
    os.chdir(input('input cwd >'))
    memmap_path = input('input x >')
    model_path = input('input model >')
    data_len = 5
    width = 960
    height = 1280
    video_mem_path = input('input target memmap >')
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    x = np.memmap(memmap_path, dtype='float32', mode='r').reshape(-1, data_len)
    y = clf.predict(x)
    
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, width, height)
    
    index_list = index_block(y, 2)
    for i, index in enumerate(tqdm(index_list)):
        data = video_mem[index]
        size = data.shape
        file_name = '{}_{}.npy'.format(i, size[0])
        arr = np.memmap(file_name, dtype='uint8', mode='w+', shape=size)
        arr[:] = data
    print(arr)