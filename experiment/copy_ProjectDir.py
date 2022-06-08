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
new_project_name = 'project_{}_{}_{}_{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
new_project_path = os.path.join(pasted_dir, new_project_name)

old_json_path = os.path.join(copied_path, 'system', 'control_dict.json')
new_json_path = os.path.join(new_project_path, 'system', 'control_dict.json')

old_cap_setting_path = os.path.join(copied_path, 'system', 'camera_settings.iccf')
new_cap_setting_path = os.path.join(new_project_path, 'system', 'camera_settings.iccf')

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
new_d['bottom'] = {}
new_d['bottom']['LightGBM_dir_path'] = os.path.join(new_project_path, 'bottom', 'LightGBM')
write_json(new_json_path, new_d)

# LightGBMモデルのコピー，パスをjsonに追加
shutil.copy2(old_side_LightGBM_path, new_side_LightGBM_path)
shutil.copy2(old_bottom_LightGBM_path, new_bottom_LightGBM_path)
new_d = read_json(new_json_path)
new_d['side']['LightGBM_model_path'] = new_side_LightGBM_path
new_d['bottom']['LightGBM_model_path'] = new_bottom_LightGBM_path
write_json(new_json_path, new_d)

# カメラの設定ファイルの書き換え
with open(old_cap_setting_path, 'r') as f:
    file_str = f.read()

old_path_list = extract_txt(file_str, '<filename>', '</filename>')

new_path_list = []
for i, old_path in enumerate(old_path_list):
    old_project_dir_name = old_path.split('\\')[-4] # 古いパスのケツから4番目のディレクトリ．今回の場合, project_y_m_d_h_s.
    new_path = old_path.replace(old_project_dir_name, new_project_name)
    new_path_list.append(new_path)

for i, old_path in enumerate(old_path_list):
    file_str = file_str.replace('<filename>'+old_path+'</filename>', '<filename>'+new_path_list[i]+'</filename>')
with open(new_cap_setting_path, 'w') as f:
    f.write(file_str)

# jsonファイルにraw_videoのパスを追記
new_d = read_json(new_json_path)
for new_path in new_path_list:
    position_dir_name = new_path.split('\\')[-3]
    raw_video_name = new_path.split('\\')[-1]
    new_d[position_dir_name]['raw_video_path'] = new_path
write_json(new_json_path, new_d)