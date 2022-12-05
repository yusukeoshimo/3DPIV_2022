import random
import numpy as np
import os

def undersampling(srcs, target, split):
    bins, thresholds = np.histogram(target, split)
    dst_lists = [[] for i in srcs]
    for i in range(bins.size):
        bool_array = (target >= thresholds[i]) & (target <= thresholds[i+1])
        sample_id = random.sample(range(np.count_nonzero(bool_array)), bins.min())
        for j, src in enumerate(srcs):
            dst_lists[j].append(src[bool_array][sample_id])
    dsts = []
    for dst_list in dst_lists:
        dst = np.concatenate(dst_list)
        dsts.append(dst)
    return dsts

if __name__ == '__main__':
    save_dir = input('input save dir >')
    x_path = input('input x path >')
    label_path = input('input y path >')
    target_id = int(input('input target id >'))

    x = np.memmap(x_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)
    label = np.memmap(label_path, np.float16, 'r').reshape(-1, 3)

    srcs = [x, label]
    target = label[:,target_id]
    x, label = undersampling(srcs, target, split=8)

    new_x = np.memmap(os.path.join(save_dir, 'UnderSampling_'+os.path.basename(x_path)), np.uint8, 'w+', shape=x.shape)
    new_label = np.memmap(os.path.join(save_dir, 'UnderSampling_'+os.path.basename(label_path)), np.float16, 'w+', shape=label.shape)
    new_x[:] = x
    new_label[:] = label