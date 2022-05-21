import numpy as np
import pickle
from tqdm import tqdm
import os
import numpy as np
from tqdm import tqdm
import math

def all_remove(arr, value):
    while value in arr:
        arr.remove(value)

def index_block(arr, value):
    classificate_list = []
    for i, label in enumerate(arr):
        if label != value:
            list_2 = []
            classificate_list.append(list_2)
        else:
            list_2.append(i)
    return [i for i in classificate_list if i != []]

def improve_index_list(index_list, fpp, fpi):
    # fpp : frame per one pulse
    # fpi : frame per iteration
    fpp = math.ceil(fpp)
    improved_list = []
    for i, indexes in enumerate(index_list):
        if i == 0:
            if abs((len(indexes)-fpp)/fpp) <= 0.1:
                start_point = indexes[0]
                improved_list.append(list(range(start_point, start_point+fpp)))
            else:
                exit('!!! my error 1 !!!')
        else:
            print(abs((indexes[0]-(start_point+fpi))/fpi))
            if abs((indexes[0]-(start_point+fpi))/fpi) <= 0.1:
                start_point = indexes[0]
                improved_list.append(list(range(start_point, start_point+fpp)))
    return improved_list

if __name__ == '__main__':
    os.chdir(input('input cwd >'))
    memmap_path = input('input x >')
    model_path = input('input model >')
    data_len = 5
    width = 960
    height = 1280
    fps = 220
    spp = 0.5 # second per pulse
    fpp = fps*spp
    fpi = fps*3
    video_mem_path = input('input target memmap >')
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    
    x = np.memmap(memmap_path, dtype='float32', mode='r').reshape(-1, data_len)
    y = clf.predict(x)
    
    video_mem = np.memmap(video_mem_path, dtype='uint8', mode='r').reshape(-1, width, height)
    
    index_list = index_block(y, 2)
    index_list = improve_index_list(index_list, fpp, fpi)
    for i, index in enumerate(tqdm(index_list)):
        data = video_mem[index]
        size = data.shape
        file_name = '{}_{}.npy'.format(i, size[0])
        arr = np.memmap(file_name, dtype='uint8', mode='w+', shape=size)
        arr[:] = data
    
    fpp = fps*spp
    print('誤分類が無いか確認中...')
    for i, indexes in enumerate(index_list):
        if len(indexes) <= fpp*0.9:
            print('誤分類：{}'.format(i))