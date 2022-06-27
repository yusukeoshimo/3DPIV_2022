import os
import numpy as np
from tqdm import tqdm

def stack_memmap(memmap_path_list, stack_memmap_path, video_height, video_width, dtype='uint8'):
    # memmapの合計サイズを調べる
    total_size = 0
    memmap_size_list = []
    for memmap_path in memmap_path_list:
        memmap = np.memmap(memmap_path, dtype=dtype, mode='r')
        total_size += int(memmap.shape[0])
        memmap_size_list.append(memmap.shape[0])
    
    # memmapをくっつける
    new_memmap = np.memmap(stack_memmap_path, dtype=dtype, mode='w+', shape=(total_size)).reshape(-1, video_height, video_width) # くっつけてできる新しいmemmap
    read_data_num = 0 # 読み込み済みのデータ数
    for memmap_path in memmap_path_list: # くっつけたいmemmapのパスが入ったリストをfor文で回す
        original_memmap = np.memmap(memmap_path, dtype=dtype, mode='r').reshape(-1, video_height, video_width) # for文で回ってきたmemmapを読み込む
        for j in tqdm(range(original_memmap.shape[0])): # 読み込んだmemmapをで画像毎にfor文で回す
            new_memmap[read_data_num+j] = original_memmap[j] # 新しいmemmapに回ってきたmemmapのデータを貼り付ける
        read_data_num += original_memmap.shape[0] # １つのmemmapをすべて貼り付け終わったら，読み込み済みのデータ数を更新

if __name__ == '__main__':
    
    # くっつけたいmemmapのパスが入ったリストを作成する
    memmap_path_list = []
    while True:
        memmap_path = input('input memmap path to stack (Press enter, and you dont be asked) >')
        if memmap_path == '':
            break
        memmap_path_list.append(memmap_path)
    
    stack_memmap_path = input('input stack memmap path >')
    
    video_height = int(input('input video height of image >'))
    video_width = int(input('intput video width of video >'))
    
    stack_memmap(memmap_path_list, stack_memmap_path, video_height, video_width)