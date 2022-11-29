from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import random
import os

def my_undersampling(srcs, target, split):
    bins, thresholds = np.histogram(target, split)
    dst_lists = [[] for i in srcs]
    for i in range(bins.size):
        bool_array = (target >= thresholds[i]) & (target <= thresholds[i+1])
        sample_id = random.sample(range(np.count_nonzero(bool_array)), bins.min()) # ここにバグありa
        for j, src in enumerate(srcs):
            dst_lists[j].append(src[bool_array][sample_id])
    dsts = []
    for dst_list in dst_lists:
        dst = np.concatenate(dst_list)
        dsts.append(dst)
    return dsts

model_path = input('input model path >')
X_path = input('input input-data >')
label_path = input('input label path >')
save_dir = input('input save dir >')
data_num = -1

model = load_model(model_path)
X0 = np.memmap(X_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)[:data_num,:,:,:2]
X1 = np.memmap(X_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)[:data_num]
label = np.memmap(label_path, np.float16, 'r').reshape(-1, 3)[:data_num,2]

X0, X1, label = my_undersampling([X0, X1, label], label, 10)
Y = model.predict([X0, X1])

# 最小二乗法
coef = np.polyfit(label, Y, 1)
a = coef[0]
b = coef[1]
print(a, b)

fig, ax = plt.subplots()
plt.scatter(label, Y, s=1, alpha=1)
ax.set_aspect('equal')
plt.plot([0,8],[a*0+b, a*8+b], color='red')
plt.savefig(os.path.join(save_dir, 'prediction_vs_label.png'))