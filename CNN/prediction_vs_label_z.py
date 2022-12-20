from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import random
import os

model_path = input('input model path >')
X_path = input('input input-data >')
label_path = input('input label path >')
save_dir = input('input save dir >')
data_num = -1
model = load_model(model_path)
X0 = np.memmap(X_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)[:data_num,:,:,:2]
X1 = np.memmap(X_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)[:data_num]
label = np.memmap(label_path, np.float16, 'r').reshape(-1, 3)[:data_num,2]

Y = model.predict([X0, X1])

mae = model.evaluate([X0, X1], label)

# 最小二乗法
coef = np.polyfit(label, Y, 1)
a = coef[0]
b = coef[1]

fig, ax = plt.subplots()
plt.title(f'a:{round(float(a),2)}, b:{round(float(b),2)}, N:{label.shape[0]}, mae:{round(mae,2)}')
plt.xlabel('label')
plt.ylabel('prediction')
plt.xlim((-1,9))
plt.ylim((-1,9))
plt.xticks(range(-1,10,1))
plt.yticks(range(-1,10,1))
plt.scatter(label, Y, s=0.5, alpha=0.1)
ax.set_aspect('equal')
plt.plot([0,8],[a*0+b, a*8+b], color='red')
plt.savefig(os.path.join(save_dir, 'prediction_vs_label.png'))