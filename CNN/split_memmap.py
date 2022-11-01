import numpy as np
import math
from sklearn.model_selection import train_test_split
import os

src_input_path = input('input src-input path >')
src_label_path = input('input src-label path >')

save_dir = input('input save dir >')

src_input = np.memmap(src_input_path, mode='r', dtype=np.uint8).reshape(-1, 32, 32, 4)
src_label = np.memmap(src_label_path, mode='r', dtype=np.float16).reshape(-1, 3)

train_input, val_tes_input, train_label, val_tes_label = train_test_split(src_input, src_label, test_size=0.1, train_size=0.9)
validation_input, test_input, validation_label, test_label = train_test_split(val_tes_input, val_tes_label, test_size=0.5, train_size=0.5)


tr_input_mem = np.memmap(os.path.join(save_dir, 'tr_input.npy'), mode='w+', dtype=train_input.dtype, shape=train_input.shape)
tr_label_mem = np.memmap(os.path.join(save_dir, 'tr_label.npy'), mode='w+', dtype=train_label.dtype, shape=train_label.shape)
val_input_mem = np.memmap(os.path.join(save_dir, 'val_input.npy'), mode='w+', dtype=validation_input.dtype, shape=validation_input.shape)
val_label_mem = np.memmap(os.path.join(save_dir, 'val_label.npy'), mode='w+', dtype=validation_label.dtype, shape=validation_label.shape)
tes_input_mem = np.memmap(os.path.join(save_dir, 'tes_input.npy'), mode='w+', dtype=test_input.dtype, shape=test_input.shape)
tes_label_mem = np.memmap(os.path.join(save_dir, 'tes_label.npy'), mode='w+', dtype=test_label.dtype, shape=test_label.shape)

tr_input_mem[:] = train_input
tr_label_mem[:] = train_label
val_input_mem[:] = validation_input
val_label_mem[:] = validation_label
tes_input_mem[:] = test_input
tes_label_mem[:] = test_label