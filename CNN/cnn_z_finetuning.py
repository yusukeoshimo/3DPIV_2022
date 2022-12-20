import os
import sys
from glob import glob
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, Dropout
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
from CNN.cnn_z import FitGen

class Objective():
    def __init__(self, model_path, tr_x_path, tr_y_path, val_x_path, val_y_path, save_dir):
        self.model_path = model_path
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
        train_input_xy = np.memmap(self.tr_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,:2]
        train_input_z = np.memmap(self.tr_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)
        train_label_z = np.memmap(self.tr_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,2]
        # 検証用データの読み込み
        validation_input_xy = np.memmap(self.val_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)[:,:,:,:2]
        validation_input_z = np.memmap(self.val_x_path, mode='r', dtype=np.uint8).reshape(-1, H, W, C)
        validation_label_z = np.memmap(self.val_y_path, mode='r', dtype=np.float16).reshape(-1, label_num)[:,2]
        
        # ハイーパーパラメータの定義
        trainable_num = trial.suggest_int('trainable_num', low=1, high=10)
        lr = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
        
        # モデルの定義
        model = keras.models.load_model(self.model_path)
        model.trainable = False
        for l in model.layers[-trainable_num:]:
            l.trainable = True
        
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
    model_path_list = glob(os.path.join(r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_z_experiment', '*.h5'))
    tr_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\training_input.npy'
    tr_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\training_label.npy'
    val_x_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_input.npy'
    val_y_path = r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\data\validation_label.npy'
    
    for i, model_path in enumerate(model_path_list):
        save_dir = os.path.join(r'c:\Users\yusuk\Documents\3dpiv_2022\cnn\cnn_z_finetuning', str(i))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        os.chdir(save_dir)
        objective = Objective(model_path, tr_x_path, tr_y_path, val_x_path, val_y_path, save_dir)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=1)
        study = optuna.create_study(pruner=pruner, study_name='cnn_z_finetuning_{}'.format(i), storage='sqlite:///cnn_z_finetuning_{}.db'.format(i), load_if_exists=True)
        study.optimize(objective, n_trials=3)