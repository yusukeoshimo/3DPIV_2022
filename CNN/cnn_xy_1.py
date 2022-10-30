import os
import sys
from unicodedata import name
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import optuna
from optuna.integration import TFKerasPruningCallback
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.my_json import read_json, write_json

class FitGen(keras.utils.Sequence):
    '''
    https://qiita.com/simonritchie/items/d7168d1d9cea9ceb6af7
    '''
    
    def __init__(self, memmap_x_0, memmap_x_1, memmap_y, batch_size):
        self.batch_size = batch_size
        self.memmap_x_0 = memmap_x_0
        self.memmap_x_1 = memmap_x_1
        self.memmap_y = memmap_y
        self.length = math.ceil(self.memmap_x_0.shape[0] / batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X_0 = self.memmap_x_0[start_idx:last_idx]
        X_1 = self.memmap_x_1[start_idx:last_idx]
        y = self.memmap_y[start_idx:last_idx]
        return {'src_input':X_0, 'dst_input':X_1}, y
    
    def __len__(self):
        return self.length

class CopyWeight(keras.callbacks.Callback):
    def __init__(self, src_layer, dst_layer):
        self.src_layer = src_layer
        self.dst_layer = dst_layer
    
    def on_train_begin(self, logs=None):
        self.dst_layer.set_weights(self.src_layer.get_weights())
    
    def on_batch_begin(self, batch, logs=None):
        self.dst_layer.set_weights(self.src_layer.get_weights())
    
    def on_train_end(self, logs=None):
        self.dst_layer.set_weights(self.src_layer.get_weights())

class Objective():
    def __init__(self, tr_x_path, tr_y_path, val_x_path, val_y_path, save_dir):
        self.tr_x_path = tr_x_path
        self.tr_y_path = tr_y_path
        self.val_x_path = val_x_path
        self.val_y_path = val_y_path
        self.save_dir = save_dir
    
    def save_best_n_model(self, json_path, model, n, trial_order, val_loss):
        model_path = os.path.join(self.save_dir, '{}_{}.h5'.format(trial_order, val_loss)) # モデルのパスを定義
        
        # jsonファイルのロード
        if not os.path.exists(json_path):
            best_n_model = {}
            write_json(json_path, best_n_model)
        else:
            best_n_model = read_json(json_path)
        
        # モデルの保存とjsonの保存
        if len(best_n_model) < n:
            model.save(model_path)
            best_n_model[model_path] = val_loss
            write_json(json_path, best_n_model)
        else:
            if val_loss < max(best_n_model.values()):
                os.remove(max(best_n_model, key=best_n_model.get)) # 一番損失値が大きいモデルを削除
                model.save(model_path) # 性能を更新したモデルを保存
                # best_n_modelを更新
                del best_n_model[max(best_n_model, key=best_n_model.get)]
                best_n_model[model_path] = val_loss
                write_json(json_path, best_n_model)
    
    def save_best_n_history(self, json_path, history, n, trial_order, val_loss):
        history_path = os.path.join(self.save_dir, '{}_{}.csv'.format(trial_order, val_loss))
        
        if not os.path.exists(json_path):
            best_n_history = {}
            write_json(json_path, best_n_history)
        else:
            best_n_history = read_json(json_path)
        
        if len(best_n_history) < n:
            pd.DataFrame(history.history).to_csv(history_path)
            best_n_history[history_path] = val_loss
            write_json(json_path, best_n_history)
        else:
            if val_loss < max(best_n_history.values()):
                os.remove(max(best_n_history, key=best_n_history.get)) # 一番損失値が大きいモデルを削除
                pd.DataFrame(history.history).to_csv(history_path)
                # best_n_historyを更新
                del best_n_history[max(best_n_history, key=best_n_history.get)]
                best_n_history[history_path] = val_loss
                write_json(json_path, best_n_history)
    
    def __call__(self, trial):
    # データの形
        H, W, C = 32, 32, 4
        label_num = 3
        
        tf.keras.backend.clear_session()
        
        # トレーニングデータの読み込み
        x_train_0 = np.memmap(self.tr_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,0].reshape(-1,H,W,1)
        x_train_1 = np.memmap(self.tr_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,1].reshape(-1,H,W,1)
        y_train = np.memmap(self.tr_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,:2]
        # 検証用データの読み込み
        x_validation_0 = np.memmap(self.val_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,0].reshape(-1,H,W,1)
        x_validation_1 = np.memmap(self.val_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,1].reshape(-1,H,W,1)
        y_validation = np.memmap(self.val_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,:2]
        
        # ハイーパーパラメータの定義
        filter_num = trial.suggest_int(name='filter_num', low=100, high=300, step=1)
        kernel_size = trial.suggest_int(name='kernel_size', low=8, high=16, step=8)
        strides = int(2**trial.suggest_int(name='strides', low=1, high=3, step=1))
        num_layer = trial.suggest_int(name='num_layer', low=4, high=6, step=1)
        mid_units = [trial.suggest_int(name='mid_units_{}'.format(i), low=50, high=300, step=1) for i in range(num_layer)]
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        
        # モデルの定義
        # src側
        src_input = Input(shape=(H,W,1), name="src_input")
        src_rescaling = Rescaling(scale=1./255, name="src_rescaling")(src_input)
        src_conv = Conv2D(filters=filter_num, kernel_size=(kernel_size, kernel_size), strides=strides, name='src_conv')(src_rescaling)
        src_BN = BatchNormalization(name='src_BN')(src_conv)
        src_relu = ReLU(name='src_Relu')(src_BN)
        src_flatten = Flatten(name='src_flatten')(src_relu)
        # dst側
        dst_input = Input(shape=(H,W,1), name="dst_input")
        dst_rescaling = Rescaling(scale=1./255, name='dst_rescaling')(dst_input)
        dst_conv = Conv2D(filters=filter_num, kernel_size=(kernel_size, kernel_size), strides=strides, name='dst_conv', trainable=False)(dst_rescaling)
        dst_BN = BatchNormalization(name='dst_BN')(dst_conv)
        dst_relu = ReLU(name='dst_Relu')(dst_BN)
        dst_flatten = Flatten(name='dst_flatten')(dst_relu)
        # src側とdst側を結合
        layer = concatenate([src_flatten, dst_flatten])
        for i in range(num_layer):
            layer = Dense(mid_units[i], name='mid_{}'.format(i))(layer)
            layer = BatchNormalization(name='BN_mid_{}'.format(i))(layer)
            layer = ReLU(name='Relu_mid_{}'.format(i))(layer)
        output = Dense(2, activation='linear', name='output')(layer)
        # 層を組み立てる
        model = keras.Model([src_input, dst_input], output)
        # 学習方法の設定
        model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
        # モデルの表示
        model.summary()
        
        # コールバックのインスタンス化
        patience = 10
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
        pruning_check = TFKerasPruningCallback(trial, monitor='val_loss')
        copy_weight = CopyWeight(model.get_layer('src_conv'), model.get_layer('dst_conv'))
        
        # 学習
        history = model.fit_generator(generator=FitGen(x_train_0, x_train_1, y_train, batch_size=50),
                                      validation_data=({'src_input':x_validation_0, 'dst_input':x_validation_1}, y_validation),
                                      epochs=500,
                                      callbacks=[early_stopping, pruning_check, copy_weight])
        
        # 検証用データの損失地を計算
        val_loss = model.evaluate({'src_input':x_validation_0, 'dst_input':x_validation_1}, y_validation, verbose=0)
        self.save_best_n_model(json_path='best_n_model.json', model=model, n=2, trial_order=trial.number, val_loss=val_loss)
        self.save_best_n_history(json_path='best_n_history.json', history=history, n=2, trial_order=trial.number, val_loss=val_loss)
        return val_loss

if __name__ == '__main__':
    tr_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\training_input.npy'
    tr_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\training_label.npy'
    val_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_input.npy'
    val_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_label.npy'
    save_dir = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_xy_1'
    
    os.chdir(save_dir)
    
    objective = Objective(tr_x_path, tr_y_path, val_x_path, val_y_path, save_dir)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=1)
    study = optuna.create_study(pruner=pruner, study_name='cnn_xy_1', storage='sqlite:///cnn_xy_1.db', load_if_exists=True)
    study.optimize(objective, n_trials=10)