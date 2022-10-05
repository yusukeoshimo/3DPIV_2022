import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def main(save_path, dir_path, width, height, turn_on_time, fps):
    frame = int(fps*turn_on_time)
    
    files = os.listdir(dir_path)
    paths = [os.path.join(dir_path, file_name) for file_name in files]
    
    df = pd.DataFrame(index=range(1, frame+1))
    for file_path in paths:
        arr = np.memmap(file_path, dtype='uint8', mode='r').reshape(-1, height, width)
        intensity_avg = np.mean(arr, axis=(1, 2))
        df[os.path.basename(file_path)] = intensity_avg
    df['mean'] = df.mean(axis=1)
    df['1_sigma'] = df.std(axis=1)
    df['1_upper'] = df['mean']+df['1_sigma']
    df['1_lower'] = df['mean']-df['1_sigma']
    df['2_sigma'] = df['1_sigma']*2
    df['2_upper'] = df['mean']+df['2_sigma']
    df['2_lower'] = df['mean']-df['2_sigma']
    
    time = df.index.values/fps
    
    fig = plt.figure()
    plt.title('intensity vs time')
    plt.xlabel('time')
    plt.ylabel('intensity')
    plt.plot(time, df['mean'], color='black', label='mean',
             marker='o', markersize=2, markerfacecolor='blue', markeredgecolor="blue")
    plt.xlim((0, turn_on_time))
    plt.ylim((0, 255))
    plt.fill_between(time, df['1_lower'], df['1_upper'], alpha=0.7, label="$1\sigma$", color='red')
    plt.fill_between(time, df['2_lower'], df['2_upper'], alpha=0.3, label="$2\sigma$", color='red')
    plt.legend(loc=1)
    fig.savefig(save_path)
    del fig

if __name__ == '__main__':
    dir_path = input('input dir path saved memmap classified by pulse state >')
    width = 1104
    height = 168
    turn_on_time = 0.5
    fps = 120
    save_path = 'fig.png'
    
    main(save_path, dir_path, width, height, turn_on_time, fps)