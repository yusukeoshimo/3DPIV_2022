import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, MaxPool2D
import tensorflow as tf
import numpy as np

# データの形
H, W, C = 32, 32, 2
label_num = 3

# データの読み込み
train_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\input.npy'
x_train = np.memmap(train_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)
x_train = np.array(x_train)
train_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\label.npy'
y_train = np.memmap(train_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)
y_train = np.array(y_train)[:,:2]

# 画像の正規化
x_train = x_train.astype(np.float16) / 255

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
model.fit(x_train, y_train, batch_size=200, epochs=200)

# 評価データに対する評価
test_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\test_input.npy'
x_test = np.memmap(test_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)
x_test = np.array(x_test)
x_test = x_test.astype(np.float16)/255
test_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\test_label.npy'
y_test = np.memmap(test_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)
y_test = np.array(y_test)[:,:2]

test_loss = model.evaluate(x_test, y_test, verbose=0)
print('test data loss:', test_loss)