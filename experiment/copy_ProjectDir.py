from asyncore import write
import subprocess
from datetime import datetime
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.my_json import read_json, write_json

copied_path = input('input project dir path you would like to copy >')
pasted_dir = os.path.join(copied_path,'..')

new_project_name = 'project_{}_{}_{}_{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
new_project_path = os.path.join(pasted_dir, new_project_name)

old_json_path = os.path.join(copied_path, 'system', 'control_dict.json')
new_json_path = os.path.join(new_project_path, 'system', 'control_dict.json')

# ディレクトリのコピー
cmd = 'xcopy /t /e "{}" "{}\\"'.format(copied_path, new_project_path)
returncode = subprocess.call(cmd)

# jsonファイルのコピー，新しい書き込み，保存
old_d = read_json(old_json_path)
new_d = {key : None for key in old_d.keys()}
new_d['project_dir'] = new_project_path
write_json(new_json_path, new_d)