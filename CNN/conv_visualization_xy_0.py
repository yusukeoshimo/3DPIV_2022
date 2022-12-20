import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
import numpy as np
import cv2

model_path = input('input model path >')
model = load_model(model_path)
model.summary()
conv_order = int(input('input order of convolution layer >'))
conv_weight = model.layers[conv_order].get_weights()[0]
conv_weight = conv_weight.transpose(3,0,1,2)

vmin, vmax = conv_weight.min(), conv_weight.max()
conv_weight = (255*(conv_weight-vmin)/(vmax-vmin)).astype(np.uint8)

fig_h = 2
fig_w = 8
fig = plt.figure()
plt.subplots_adjust(wspace=0.3, hspace=0)
for i in range(fig_w):
    plt.subplot(fig_h,fig_w,i+1)
    plt.imshow(cv2.cvtColor(conv_weight[i,:,:,0], cv2.COLOR_BGR2RGB))
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.title(f'kernel {i+1}', fontsize=8)
    if i == 0:
        plt.ylabel('channel 1', fontsize=8)
for i in range(fig_w):
    plt.subplot(fig_h,fig_w,fig_w+i+1)
    plt.imshow(cv2.cvtColor(conv_weight[i,:,:,1], cv2.COLOR_BGR2RGB))
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    if i == 0:
        plt.ylabel('channel 2', fontsize=8)
plt.tight_layout()

plt.show()