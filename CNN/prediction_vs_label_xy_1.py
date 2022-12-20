from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import random
import os

model_path = input('input model path >')
X_path = input('input input-data >')
label_path = input('input label path >')
save_dir = input('input save dir >')
x_or_y = int(input('input value (x->0, y->1) >'))

model = load_model(model_path)
X0 = np.memmap(X_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)[:,:,:,0]
X1 = np.memmap(X_path, np.uint8, 'r',).reshape(-1, 32, 32, 4)[:,:,:,1]
label = np.memmap(label_path, np.float16, 'r').reshape(-1, 3)[:,:2]

Y = model.predict([X0,X1])

maes = (np.sum(abs(label-Y), axis=0))/label.shape[0]

# 最小二乗法
coef = np.polyfit(label[:,x_or_y], Y[:,x_or_y], 1)
a = coef[0]
b = coef[1]

fig, ax = plt.subplots()
plt.title(f'a:{round(float(a),2)}, b:{round(float(b),2)}, N:{label.shape[0]}, mae:{round(float(maes[x_or_y]),4)}')
plt.xlabel('label')
plt.ylabel('prediction')
plt.xlim((-9,9))
plt.ylim((-9,9))
plt.xticks(range(-9,10,1))
plt.yticks(range(-9,10,1))
plt.scatter(label[:,x_or_y], Y[:,x_or_y], s=0.5, alpha=0.1)
ax.set_aspect('equal')
plt.plot([-8,8],[a*(-8)+b, a*8+b], color='red')
plt.savefig(os.path.join(save_dir, f'prediction_vs_label_{x_or_y}.png'))