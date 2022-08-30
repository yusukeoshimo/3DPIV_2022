from re import X
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, MaxPool2D
import tensorflow as tf
import numpy as np
import math

class FitGen(keras.utils.Sequence):
    '''
    https://qiita.com/simonritchie/items/d7168d1d9cea9ceb6af7
    '''

    def __init__(self, memmap_x, memmap_y, batch_size):
        self.batch_size = batch_size
        self.memmap_x = memmap_x
        self.memmap_y = memmap_y
        self.length = math.ceil(self.memmap_x.shape[0] / batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.memmap_x[start_idx:last_idx]
        y = self.memmap_y[start_idx:last_idx]
        return X, y
    
    def __len__(self):
        return self.length

# データの形
H, W, C = 32, 32, 2
label_num = 3

# データの読み込み
train_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\input.npy'
x_train = np.memmap(train_x_path, mode='r', dtype=np.float16).reshape(-1, H, W, C)
train_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\label.npy'
y_train = np.memmap(train_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,:2]

# モデルの定義
inputs = tf.keras.layers.Input(shape=(H,W,C), name="inputs")
conv1 = Conv2D(filters=32, kernel_size=(3, 3))(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = ReLU()(conv1)
conv1 = MaxPool2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3))(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = ReLU()(conv2)
conv2 = MaxPool2D((2, 2))(conv2)
conv3 = Conv2D(64, (3, 3))(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = ReLU()(conv3)
conv3_flat = Flatten()(conv3)
fc1 = Dense(64,activation='relu')(conv3_flat)
outputs = Dense(2,activation='linear')(fc1)

model = keras.Model(inputs, outputs)

# 学習方法の設定
model.compile(optimizer='adam',loss='mae')

model.summary()
model.fit_generator(generator=FitGen(x_train, y_train, batch_size=50), epochs=50)

# 評価データに対する評価
test_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\test_input.npy'
x_test = np.memmap(test_x_path, mode='r', dtype=np.float16).reshape(-1, H, W, C)
x_test = np.array(x_test)
test_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\test_label.npy'
y_test = np.memmap(test_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)
y_test = np.array(y_test)[:,:2]
test_loss = model.evaluate(x_test, y_test, verbose=0)
print('test data loss:', test_loss)