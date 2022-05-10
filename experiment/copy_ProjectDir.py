import subprocess
from datetime import datetime
import os

copied_path = input('input project dir path you would like to copy >')
pasted_dir = input('input dir path you would like to paste >')

new_project_name = 'project_{}_{}_{}_{}_{}_{}'.format(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second)
new_project_path = os.path.join(pasted_dir, new_project_name)

# ディレクトリのコピー
cmd = 'xcopy /t /e "{}" "{}\\"'.format(copied_path, new_project_path)
returncode = subprocess.call(cmd)