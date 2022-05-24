import os
import json

def apend_json(json_path, key, value):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            d = json.load(f)
    else:
        d = {}
    
    d[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)

def read_json(json_path):
    with open(json_path, 'r') as f:
        d = json.load(f)
    return d

if __name__ == '__main__':
    cwd = input('input project dir >')
    os.chdir(cwd)
    dict_path = os.path.join('system', 'control_dict.json')
    d = {'test3':None}
    apend_json(dict_path, 'test3', d)
    