import os
import optuna
import pandas as pd

save_dir = input('input dir saved database >')
study_name = input('input study name >')
storage_name = input('input storage name >')

os.chdir(save_dir)

study = optuna.load_study(study_name=study_name, storage='sqlite:///{}.db'.format(storage_name))

df = study.trials_dataframe()
df = df.sort_values('value')
print(df[['number', 'value', 'params_model_path']])