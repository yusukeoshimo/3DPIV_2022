import numpy as np
import os
import cv2
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from util.my_json import read_json, write_json



if __name__ == '__main__':
    project_dir_path = input('input project dir >')
    position = input('input position (side or bottom) >')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    control_dict = read_json(json_path)
    cwd = control_dict[position]['data_2_memmap_dir_path'] # memmapが保存されているディレクトリ
    os.chdir(cwd)
    
    save_dir = control_dict[position]['data_2_dir_path'] # 作成した動画を保存するディレクトリ
    frame_interval = 2 # 取得フレームがどれだけ離れたフレームか．1~
    cut_edge = 2 # 動画の両端をいくら切り捨てるか．0~
    save_path = os.path.join(save_dir, 'FrameInterval_{}.avi'.format(frame_interval))
    height = control_dict[position]['video_height'] # フレームの高さ
    width = control_dict[position]['video_width'] # フレームの幅
    fps = 1 # 作成する動画のfps
    first_pulse = control_dict[position]['first_pulse'] # 最初に点灯するレーザー．1 or 2
    if first_pulse == 1:
        extract_0 = 0+cut_edge # 取得するフレーム．
        extract_1 = extract_0+frame_interval # 取得するフレーム
    else:
        extract_1 = -1-cut_edge # 取得するフレーム
        extract_0 = extract_1-frame_interval # 取得するフレーム
    
    # jsonに書き込み
    control_dict = read_json(json_path)
    control_dict[position]['frame_interval'] = frame_interval
    control_dict[position]['cut_edge'] = cut_edge
    write_json(json_path, control_dict)
    
    file_dict = {int(file_name.split('_')[0]) : file_name for file_name in os.listdir(cwd)} # {xxx:'xxx_yyy.npy', ....}
    
    fourcc = cv2.VideoWriter_fourcc('Y','8','0','0')
    video = cv2.VideoWriter(save_path,fourcc,fps,(width, height))
    for i in tqdm(range(len(file_dict))):
        arr = np.memmap(file_dict[i], dtype='uint8', mode='r').reshape(-1, height, width)
        for frame in [arr[extract_0], arr[extract_1]]:
            frame = np.repeat(frame, 3).reshape(height,width,3).astype(np.uint8)
            video.write(frame)
    video.release()