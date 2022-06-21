import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

if __name__ == '__main__':
    # dir_path = input('input dir path saved memmap >')
    dir_path = input('input dir path saved data_2_memmap >')
    files = os.listdir(dir_path)
    paths = [os.path.join(dir_path, file_name) for file_name in files]
    
    background_list = []
    for file_path in paths:
        arr = np.memmap(file_path, dtype='uint8', mode='r').reshape(-1, 960, 1280)[1:-2] # レーザーが点灯しきっていないエッジをカットする
        background = np.amin(arr, axis=0)
        background_list.append(background)
    background = np.amin(np.array(background_list), axis=0)
    
    intensity_avg_list = []
    for file_path in paths:
        arr = np.memmap(file_path, dtype='uint8', mode='r').reshape(-1, 960, 1280)[1:-2] # レーザーが点灯しきっていないエッジをカットする
        intensity_avg = np.mean(arr, axis=0)
        intensity_avg_list.append(intensity_avg)
    intensity_avg = np.mean(np.array(intensity_avg_list), axis=0)
    intensity_avg = intensity_avg-background # 背景抜き
    intensity_avg = intensity_avg/np.amax(intensity_avg) # 正規化
    
    
    # グラフを描写
    fig, ax = plt.subplots()
    ax.set_title('intensity vs location')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    c = ax.contourf(np.flipud(intensity_avg), 20, cmap="jet")
    fig.colorbar(c)
    plt.show()