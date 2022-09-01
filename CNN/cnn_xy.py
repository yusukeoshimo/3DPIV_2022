import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, MaxPool2D
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import optuna

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

def create_model(mid_units):
    H, W, C = 32, 32, 2
    label_num = 3
    # モデルの定義
    inputs = tf.keras.layers.Input(shape=(H,W,C), name="inputs")
    conv1 = Conv2D(filters=200, kernel_size=(8, 8), strides=4)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv3_flat = Flatten()(conv1)
    fc1 = Dense(200,activation='relu')(conv3_flat)
    fc2 = Dense(mid_units[0], activation='relu')(fc1)
    outputs = Dense(2, activation='linear')(fc2)
    
    model = keras.Model(inputs, outputs)
    
    # 学習方法の設定
    model.compile(optimizer='adam',loss='mae')
    
    model.summary()
    
    return model

def objective(trial):
    # データの形
    H, W, C = 32, 32, 2
    label_num = 3

    # トレーニングデータの読み込み
    train_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\input.npy'
    x_train = np.memmap(train_x_path, mode='r', dtype=np.float16).reshape(-1, H, W, C)
    train_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\label.npy'
    y_train = np.memmap(train_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,:2]
    # 検証用データの読み込み
    test_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\test_input.npy'
    x_validation = np.memmap(test_x_path, mode='r', dtype=np.float16).reshape(-1, H, W, C)
    test_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\test_label.npy'
    y_validation = np.memmap(test_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,:2]

    # ハイーパーパラメータの定義
    num_layer = 1
    mid_unit_1 = trial.suggest_int(name='mid_units_1', low=50, high=300, step=1)
    
    # モデルの定義
    inputs = tf.keras.layers.Input(shape=(H,W,C), name="inputs")
    conv1 = Conv2D(filters=200, kernel_size=(8, 8), strides=4)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv3_flat = Flatten()(conv1)
    fc1 = Dense(200,activation='relu')(conv3_flat)
    fc2 = Dense(mid_unit_1, activation='relu')(fc1)
    outputs = Dense(2, activation='linear')(fc2)
    # 層を組み立てる
    model = keras.Model(inputs, outputs)
    # 学習方法の設定
    model.compile(optimizer='adam',loss='mae')
    # モデルの表示
    model.summary()
    
    # 学習
    patience = 5
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    model.fit_generator(generator=FitGen(x_train, y_train, batch_size=50),validation_data=(x_validation, y_validation), epochs=50, callbacks=[early_stopping])
    
    # 検証用データの損失地を計算
    test_loss = model.evaluate(x_validation, y_validation, verbose=0)
    return test_loss

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=3)