import numpy as np
import os
from tqdm import tqdm

save_dir = input('input save dir >')
x_path = input('input input-data path >')
label_path = input('input label path >')

x = np.memmap(x_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)
label = np.memmap(label_path, np.float16, 'r').reshape(-1, 3)
new_x = np.memmap(os.path.join(save_dir, 'shuffle_'+os.path.basename(x_path)), np.uint8, 'w+', shape=x.shape)
new_label = np.memmap(os.path.join(save_dir, 'shuffle_'+os.path.basename(label_path)), np.float16, 'w+', shape=label.shape)

order = np.arange(label.shape[0])
np.random.shuffle(order)

for i in tqdm(range(label.shape[0])):
    new_x[i] = x[order[i]]
    new_label[i] = label[order[i]]