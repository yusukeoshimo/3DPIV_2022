import os
import sys
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Dropout, Input, Average, Concatenate
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import optuna
from optuna.integration import KerasPruningCallback
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.my_json import read_json, write_json

class FitGen(keras.utils.Sequence):
    '''
    https://qiita.com/simonritchie/items/d7168d1d9cea9ceb6af7
    '''
    
    def __init__(self, train_input_xy, train_input_z, train_label_z, batch_size):
        self.batch_size = batch_size
        self.train_input_xy = train_input_xy
        self.train_input_z = train_input_z
        self.train_label_z = train_label_z
        self.length = math.ceil(self.train_input_xy.shape[0] / batch_size)
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X_0 = self.train_input_xy[start_idx:last_idx]
        X_1 = self.train_input_z[start_idx:last_idx]
        y = self.train_label_z[start_idx:last_idx]
        return {'input_xy':X_0, 'input_z':X_1}, y
    
    def __len__(self):
        return self.length

class Objective():
    def __init__(self, tr_x_path, tr_y_path, val_x_path, val_y_path, save_dir, xy_estimator_0_path, xy_estimator_1_path, xy_estimator_2_path, xy_estimator_3_path, xy_estimator_4_path):
        self.tr_x_path = tr_x_path
        self.tr_y_path = tr_y_path
        self.val_x_path = val_x_path
        self.val_y_path = val_y_path
        self.save_dir = save_dir
        self.xy_estimator_0_path = xy_estimator_0_path
        self.xy_estimator_1_path = xy_estimator_1_path
        self.xy_estimator_2_path = xy_estimator_2_path
        self.xy_estimator_3_path = xy_estimator_3_path
        self.xy_estimator_4_path = xy_estimator_4_path
    
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
        train_input_xy = np.memmap(self.tr_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,:2]
        train_input_z = np.memmap(self.tr_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)
        train_label_z = np.memmap(self.tr_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,2]
        # 検証用データの読み込み
        validation_input_xy = np.memmap(self.val_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,:2]
        validation_input_z = np.memmap(self.val_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)
        validation_label_z = np.memmap(self.val_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,2]

        # xy予測モデルの読み込み
        xy_estimator_0 = keras.models.load_model(self.xy_estimator_0_path)
        xy_estimator_1 = keras.models.load_model(self.xy_estimator_1_path)
        xy_estimator_2 = keras.models.load_model(self.xy_estimator_2_path)
        xy_estimator_3 = keras.models.load_model(self.xy_estimator_3_path)
        xy_estimator_4 = keras.models.load_model(self.xy_estimator_4_path)
        # モデルに固有の名前を与える
        xy_estimator_0._name = 'xy_estimator_0'
        xy_estimator_1._name = 'xy_estimator_1'
        xy_estimator_2._name = 'xy_estimator_2'
        xy_estimator_3._name = 'xy_estimator_3'
        xy_estimator_4._name = 'xy_estimator_4'
        # xy予測モデルのをtrainableをFalseにする
        xy_estimator_0.trainable = False
        xy_estimator_1.trainable = False
        xy_estimator_2.trainable = False
        xy_estimator_3.trainable = False
        xy_estimator_4.trainable = False
        
        # ハイーパーパラメータの定義
        filter_num = trial.suggest_int(name='filter_num', low=100, high=300, step=1)
        kernel_size = trial.suggest_int(name='kernel_size', low=8, high=16, step=8)
        strides = int(2**trial.suggest_int(name='strides', low=1, high=3, step=1))
        num_layer = trial.suggest_int(name='num_layer', low=4, high=6, step=1)
        mid_units = [trial.suggest_int(name='mid_units_{}'.format(i), low=50, high=300, step=1) for i in range(num_layer)]
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        # モデルの定義
        # xyの部分
        input_xy = Input(shape=(H, W, 2), name='input_xy')
        layer_xy_0 = xy_estimator_0(input_xy)
        layer_xy_1 = xy_estimator_1(input_xy)
        layer_xy_2 = xy_estimator_2(input_xy)
        layer_xy_3 = xy_estimator_3(input_xy)
        layer_xy_4 = xy_estimator_4(input_xy)
        output_xy = Average()([layer_xy_0, layer_xy_1, layer_xy_2, layer_xy_3, layer_xy_4])
        # zの部分
        input_z = Input(shape=(H, W, 4), name='input_z')
        layer = Rescaling(scale=1./255)(input_z)
        layer = Conv2D(filters=filter_num, kernel_size=(kernel_size, kernel_size), strides=strides, name='conv')(layer)
        layer = BatchNormalization(name='BN_conv')(layer)
        layer = ReLU(name='Relu_conv')(layer)
        layer = Flatten(name='flatten')(layer)
        layer = Concatenate()([layer, output_xy])
        for i in range(num_layer):
            layer = Dense(mid_units[i], name='mid_{}'.format(i))(layer)
            layer = BatchNormalization(name='BN_mid_{}'.format(i))(layer)
            layer = ReLU(name='Relu_mid_{}'.format(i))(layer)
        output_z = Dense(1, activation='linear', name='output_z')(layer)
        # 層を組み立てる
        model = keras.Model([input_xy, input_z], output_z)
        # 学習方法の設定
        model.compile(optimizer=Adam(learning_rate=lr), loss='mae')
        # モデルの表示
        model.summary()
        # 学習
        patience = 10
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
        history = model.fit_generator(generator=FitGen(train_input_xy, train_input_z, train_label_z, batch_size=50),
                                      validation_data=({'input_xy':validation_input_xy, 'input_z':validation_input_z}, validation_label_z),
                                      epochs=500,
                                      callbacks=[early_stopping, KerasPruningCallback(trial, monitor='val_loss')])
        
        # 検証用データの損失地を計算
        val_loss = model.evaluate({'input_xy':validation_input_xy, 'input_z':validation_input_z}, validation_label_z, verbose=0)
        self.save_best_n_model(json_path='best_n_model.json', model=model, n=2, trial_order=trial.number, val_loss=val_loss)
        self.save_best_n_history(json_path='best_n_history.json', history=history, n=2, trial_order=trial.number, val_loss=val_loss)
        return val_loss

if __name__ == '__main__':
    tr_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\training_input.npy'
    tr_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\training_label_abs.npy'
    val_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_input.npy'
    val_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_label_abs.npy'
    save_dir = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_z'
    xy_estimator_0_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_xy_0\11_0.07718148082494736.h5'
    xy_estimator_1_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_xy_0\12_0.07790252566337585.h5'
    xy_estimator_2_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_xy_0\14_0.07766833901405334.h5'
    xy_estimator_3_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_xy_0\15_0.07880500704050064.h5'
    xy_estimator_4_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_xy_0\16_0.07958551496267319.h5'
    
    os.chdir(save_dir)
    
    objective = Objective(tr_x_path, tr_y_path, val_x_path, val_y_path, save_dir,xy_estimator_0_path, xy_estimator_1_path, xy_estimator_2_path, xy_estimator_3_path, xy_estimator_4_path)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=1)
    study = optuna.create_study(pruner=pruner, study_name='cnn_z', storage='sqlite:///cnn_xy_0.db', load_if_exists=True)
    study.optimize(objective, n_trials=30)