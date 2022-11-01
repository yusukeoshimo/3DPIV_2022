from tkinter.tix import Tree
import numpy as np
import cv2
from sqlalchemy import false, true
from tqdm import tqdm
import os
import itertools

input_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_input.npy'
label_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_label.npy'
save_dir = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data'
# input_path = input('input input-memmap path >')
# label_path = input('input src_label-memmap path >')
# save_dir = input('input save dir >')

src_input = np.memmap(input_path, mode='r', dtype=np.uint8).reshape(-1, 32, 32, 4)
src_label = np.memmap(label_path, mode='r', dtype=np.float16).reshape(-1, 3)

dst_input = np.memmap(os.path.join(save_dir, 'increase_input.npy'), mode='w+', dtype=src_input.dtype, shape=(8*src_input.shape[0], src_input.shape[1], src_input.shape[2], src_input.shape[3]))
dst_label = np.memmap(os.path.join(save_dir, 'increase_label.npy'), mode='w+', dtype=src_label.dtype, shape=(8*src_label.shape[0], src_label.shape[1]))


product_list = list(itertools.product([False, True], repeat=3))

for i, bool_lsit in enumerate(product_list):
    rl_bool, ud_bool, revs_bool = bool_lsit
    src_input = np.memmap(input_path, mode='r', dtype=np.uint8).reshape(-1, 32, 32, 4)
    src_label = np.memmap(label_path, mode='r', dtype=np.float16).reshape(-1, 3)
    if rl_bool:
        src_input = np.flip(src_input, axis=2)
        src_label = np.stack((-src_label[:,0], src_label[:,1], src_label[:,2]), axis=-1)
    if ud_bool:
        src_input = np.flip(src_input, axis=1)
        src_label = np.stack((src_label[:,0], -src_label[:,1], src_label[:,2]), axis=-1)
    if revs_bool:
        src_input = np.stack((src_input[:,:,:,1], src_input[:,:,:,0], src_input[:,:,:,2], src_input[:,:,:,3]), axis=-1)
        src_label = np.stack((-src_label[:,0], -src_label[:,1], -src_label[:,2]), axis=-1)
    dst_input[i*len(src_input):(i+1)*len(src_input)] = src_input
    dst_label[i*len(src_label):(i+1)*len(src_label)] = src_label

dst_input = np.memmap(os.path.join(save_dir, 'increase_input.npy'), mode='r', dtype=src_input.dtype, shape=(8*src_input.shape[0], src_input.shape[1], src_input.shape[2], src_input.shape[3]))
dst_label = np.memmap(os.path.join(save_dir, 'increase_label.npy'), mode='r', dtype=src_label.dtype, shape=(8*src_label.shape[0], src_label.shape[1]))

for i in range(8):
    cv2.imwrite(f'{product_list[i][0]}_{product_list[i][1]}_{product_list[i][2]}.bmp', dst_input[i*1000,:,:,0])
    print(dst_label[i*1000,:])