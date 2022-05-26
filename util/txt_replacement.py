from dataclasses import replace
import re
import os

# ある単語とある単語の間の単語を抽出する関数 （注）文末にある単語を選択するときは back= '\n'
def extract_txt(txt, front, back):
    ptn = '{}(.*){}'.format(front,back)
    match_str_list = re.findall(ptn, txt) # 二次元のリスト
    return match_str_list
    
if __name__ == '__main__':
    setting_file_path = r'c:\Users\yusuk\Desktop\3DPIV_2022\project\0_camera_settings\camera_settings.iccf'
    
    with open(setting_file_path, 'r') as f:
        old_file_str = f.read()
    
    old_path_list = extract_txt(old_file_str, '<filename>', '</filename>')
    new_project_dir_name = 'project_22222222222'
    new_path_list = []
    for i, old_path in enumerate(old_path_list):
        old_project_dir_name = old_path.split('\\')[-4]
        new_path = old_path.replace(old_project_dir_name, new_project_dir_name)
        new_path_list.append(new_path)
    
    for i, old_path in enumerate(old_path_list):
        new_file_str = old_file_str.replace('<filename>'+old_path+'</filename>', '<filename>'+new_path_list[i]+'</filename>')
        print('{} -> {}'.format(old_path, new_path_list[i]))
    
    with open(setting_file_path, 'w') as f:
        f.write(new_file_str)