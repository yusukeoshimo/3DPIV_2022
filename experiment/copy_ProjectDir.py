from asyncore import write
import subprocess
from datetime import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.my_json import read_json, write_json
from util.txt_replacement import extract_txt
from util import winpath
import shutil

copied_path = input('input project dir path you would like to copy >')
pasted_dir = winpath.join(*copied_path.split('\\')[:-1])
old_project_name = copied_path.split('\\')[-1]
new_project_name = 'project_{}_{}_{}_{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
new_project_path = os.path.join(pasted_dir, new_project_name)

old_json_path = os.path.join(copied_path, 'system', 'control_dict.json')
new_json_path = os.path.join(new_project_path, 'system', 'control_dict.json')

old_side_LightGBM_path = os.path.join(copied_path, 'side', 'LightGBM', 'my_LightGBM.pkl')
old_bottom_LightGBM_path = os.path.join(copied_path, 'bottom', 'LightGBM', 'my_LightGBM.pkl')
new_side_LightGBM_path = os.path.join(new_project_path, 'side', 'LightGBM', 'my_LightGBM.pkl')
new_bottom_LightGBM_path = os.path.join(new_project_path, 'bottom', 'LightGBM', 'my_LightGBM.pkl')

# ディレクトリのコピー
cmd = 'xcopy /t /e "{}" "{}\\"'.format(copied_path, new_project_path)
returncode = subprocess.call(cmd)

# jsonファイルのコピー，リセット，新しい書き込み，保存
old_d = read_json(old_json_path)
new_d = {key : None for key in old_d.keys()}
new_d['project_dir_path'] = new_project_path
new_d['side'] = {}
new_d['side']['LightGBM_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM')
new_d['side']['0_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM', '0')
new_d['side']['1_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM', '1')
new_d['side']['2_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM', '2')
new_d['side']['appending_0_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM', 'appending_0')
new_d['side']['appending_1_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM', 'appending_1')
new_d['side']['appending_2_dir_path'] = os.path.join(new_project_path, 'side', 'LightGBM', 'appending_2')
new_d['side']['data_2_dir_path'] = os.path.join(new_project_path, 'side', 'data_2')
new_d['side']['data_2_memmap_dir_path'] = os.path.join(new_project_path, 'side', 'data_2', 'memmap')
new_d['bottom'] = {}
new_d['bottom']['LightGBM_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM')
new_d['bottom']['0_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM', '0')
new_d['bottom']['1_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM', '1')
new_d['bottom']['2_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM', '2')
new_d['bottom']['appending_0_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM', 'appending_0')
new_d['bottom']['appending_1_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM', 'appending_1')
new_d['bottom']['appending_2_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM', 'appending_2')
new_d['bottom']['data_2_dir_path'] = os.path.join(new_project_path, 'bottom', 'data_2')
new_d['bottom']['data_2_memmap_dir_path'] = os.path.join(new_project_path, 'bottom', 'data_2', 'memmap')
write_json(new_json_path, new_d)

# LightGBMモデルのコピー，パスをjsonに追加
shutil.copy2(old_side_LightGBM_path, new_side_LightGBM_path)
shutil.copy2(old_bottom_LightGBM_path, new_bottom_LightGBM_path)
new_d = read_json(new_json_path)
new_d['side']['LightGBM_model_path'] = new_side_LightGBM_path
new_d['bottom']['LightGBM_model_path'] = new_bottom_LightGBM_path
write_json(new_json_path, new_d)

# LightGBMモデルの学習データをコピー
old_d = read_json(old_json_path)
new_d = read_json(new_json_path)

new_d['side']['learning_input_path'] = old_d['side']['learning_input_path'].replace(old_project_name, new_project_name)
shutil.copy2(old_d['side']['learning_input_path'], new_d['side']['learning_input_path'])

new_d['bottom']['learning_input_path'] = old_d['bottom']['learning_input_path'].replace(old_project_name, new_project_name)
shutil.copy2(old_d['bottom']['learning_input_path'], new_d['bottom']['learning_input_path'])

new_d['side']['learning_label_path'] = old_d['side']['learning_label_path'].replace(old_project_name, new_project_name)
shutil.copy2(old_d['side']['learning_label_path'], new_d['side']['learning_label_path'])

new_d['bottom']['learning_label_path'] = old_d['bottom']['learning_label_path'].replace(old_project_name, new_project_name)
shutil.copy2(old_d['bottom']['learning_label_path'], new_d['bottom']['learning_label_path'])

write_json(new_json_path, new_d)

recalib_bool = input('recalib? (y or n) >')
for position in ['side', 'bottom']:
    if recalib_bool == 'y':
        recalib_bool = True
        old_calibration_path = os.path.join(copied_path, position, 'calibration')
        new_calibration_path = os.path.join(new_project_path, position, 'calibration')
        # di5ファイルの書き換えと保存
        with open(os.path.join(old_calibration_path, 'calibration.di5')) as f:
            txt = f.read()
        file_path = extract_txt(txt, '<ProjectPath>', '</ProjectPath>')[0]
        file_list = file_path.split('\\')
        file_list[-3] = new_project_name
        txt.replace(file_path, winpath.join(*file_list))
        with open(os.path.join(new_calibration_path, 'calibration.di5'), mode='w') as f:
            f.write(txt)
        # 校正のための参照ファイルをコピー
        shutil.copy2(os.path.join(old_calibration_path, 'Cam01', 'Calib01', 'CalibAtOnce.xml'), os.path.join(new_calibration_path, 'Cam01', 'Calib01', 'CalibAtOnce.xml'))
        shutil.copy2(os.path.join(old_calibration_path, 'Cam01', 'Calib01', 'CalibrationParameters.xml'), os.path.join(new_calibration_path, 'Cam01', 'Calib01', 'CalibrationParameters.xml'))
        shutil.copy2(os.path.join(old_calibration_path, 'Cam01', 'Calib01', 'MaskImage.png'), os.path.join(new_calibration_path, 'Cam01', 'Calib01', 'MaskImage.png'))
        shutil.copy2(os.path.join(old_calibration_path, 'Cam01', 'Calib01', 'MaskObject.xml'), os.path.join(new_calibration_path, 'Cam01', 'Calib01', 'MaskObject.xml'))
        shutil.copy2(os.path.join(old_calibration_path, 'Cam01', 'Take01', 'ExtractedGrids.xml'), os.path.join(new_calibration_path, 'Cam01', 'Take01', 'ExtractedGrids.xml'))
        
    elif recalib_bool == 'n':
        recalib_bool = False
        shutil.copy2(os.path.join(copied_path, position, 'extracted_data', 'grid.csv'), os.path.join(new_project_path, position, 'extracted_data', 'grid.csv'))

    new_d = read_json(new_json_path)
    new_d['recalib_bool'] = recalib_bool
    write_json(new_json_path, new_d)