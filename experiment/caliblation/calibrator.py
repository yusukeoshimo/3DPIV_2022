import os
import cv2
import numpy as np
import pandas as pd
import re
import json
from PIL import Image
import glob
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..\..'))
from util.my_json import read_json, apend_json, write_json

# ある単語とある単語の間の単語をjsonに保存する関数 （注）文末にある単語を選択するときは back= '\n'
def txt2json(txt_path, front, back, mode, json_path, data_name):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ptn = '{}(.*){}'.format(front,back)
            if front in line:
                data = re.findall(ptn, line)[0]
    if mode == 'w':
        with open(json_path, mode) as f:
            json.dump({data_name:data}, f, indent=4)
    if mode == 'a':
        with open(json_path, 'r') as f:
            d = json.load(f)
            d[data_name] = data
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=4)

# ある文字列とある文字列の間にあるテキストをcsvに変換する関数
def txt2csv(txt_path, start_str, end_str, save_path='grid.csv'):
    with open(txt_path,'r+') as f:
        lines = f.readlines()
        lines = [re.sub(r', ', ',', line) for line in lines]
        for i, line in enumerate(lines):
            if start_str in line:
                start_id = i+1
            if end_str in line:
                end_id = i
        grid = lines[start_id:end_id]
    if os.path.exists(save_path):
        os.remove(save_path)
    with open (save_path, 'a') as f:
        for i in grid:
            f.write(i)

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

class calibrator():
    
    def __init__(self, grid_len_phys, grid_len_pix, csv_path):
        self.grid_len_phys = grid_len_phys
        self.grid_len_pix = grid_len_pix
        self.csv_path = csv_path
    
    
    def pre_processing(self):
        # 左上に原点を合わせる
        df = pd.read_csv(self.csv_path)
        df['Given Xmm'] = df['Given Xmm']-df['Given Xmm'].min(0)
        df['Given Ymm'] = df['Given Ymm']-df['Given Ymm'].min(0)
        
        # csvにgrid_stepを追加
        df['grid_step_x'] = df['Given Xmm']/self.grid_len_phys
        df['grid_step_y'] = df['Given Ymm']/self.grid_len_phys
        
        # transposeした時の座標をcsvに追加
        df['p_trans_x'] = df['grid_step_x']*self.grid_len_pix
        df['p_trans_y'] = df['grid_step_y']*self.grid_len_pix
        df.to_csv(self.csv_path)
        
    def transpose(self, image):
        
        mat = Image.new('L', (int(pd.read_csv(self.csv_path).loc[:, ['p_trans_x']].max(0)), int(pd.read_csv(self.csv_path).loc[:, ['p_trans_y']].max(0))), 0)
        
        origin_list = pd.read_csv(self.csv_path).loc[:,['grid_step_x', 'grid_step_y']].values.tolist()
        df = pd.read_csv(self.csv_path)
        for origin in origin_list:
        
            grid_step = [origin, [origin[0]+1, origin[1]], [origin[0]+1, origin[1]+1], [origin[0], origin[1]+1]]
            id_of_grid = df.index[(df['grid_step_x'] == grid_step[0][0])&(df['grid_step_y'] == grid_step[0][1])|
                                  (df['grid_step_x'] == grid_step[1][0])&(df['grid_step_y'] == grid_step[1][1])|
                                  (df['grid_step_x'] == grid_step[2][0])&(df['grid_step_y'] == grid_step[2][1])|
                                  (df['grid_step_x'] == grid_step[3][0])&(df['grid_step_y'] == grid_step[3][1])].tolist()
            
            if len(id_of_grid) == 4:
                p_original = df.loc[id_of_grid,['Given Xpix', 'Given Ypix']].values.tolist()
                p_trans = (df.loc[id_of_grid,['p_trans_x', 'p_trans_y']].values-np.array(origin)*self.grid_len_pix).tolist()
                
                p_original = np.float32(p_original)
                p_trans = np.float32(p_trans)
                M = cv2.getPerspectiveTransform(p_original, p_trans)
                i_trans = cv2.warpPerspective(image, M, (self.grid_len_pix, self.grid_len_pix))
                
                p_trans_origin = (int(origin[0]*self.grid_len_pix),int(origin[1]*self.grid_len_pix))
                
                mat.paste(Image.fromarray(i_trans), p_trans_origin)
        mat = pil2cv(mat)
        return mat

def video_processing(original_video_path, calibed_video_path, f, output_size):
    def wrapper(*args, **kwargs):
        video_path = original_video_path
        cap = cv2.VideoCapture(video_path) #読み込む動画のパス
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('utf-8')) #mp4フォーマット
        video = cv2.VideoWriter(calibed_video_path, fourcc, fps, output_size) #書き込み先のパス、フォーマット、fps、サイズ

        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            
            # ここに処理を記入********************************************
            frame = f(frame, *args, **kwargs)
            frame = np.zeros((output_size[1],output_size[0],3), np.uint8)+frame.reshape(frame.shape[0],frame.shape[1],1)
            # **********************************************************
            
            video.write(frame)
            
        cap.release()
        video.release()
    return wrapper


if __name__ == '__main__':
    
    project_dir_path = input('input project dir path >')
    position_dir_name = input('input bottom or side >')
    os.chdir(os.path.join(project_dir_path, position_dir_name))
    
    calib_path = os.path.join('calibration','Cam01','Calib01','Calibration.cal')
    json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')
    csv_path = os.path.join('extracted_data','grid.csv')
    orignal_image_path = os.path.join('calib_board', 'off.bmp')
    calibed_image_path = os.path.join('calibrated_image','mapped_{}.bmp'.format(os.path.splitext(os.path.basename(orignal_image_path))[0]))
    orignal_video_path = glob.glob(os.path.join('raw_video', '*.avi'))[0]
    calibed_video_path = os.path.join('calibrated_video','mapped_{}.avi'.format(os.path.splitext(os.path.basename(orignal_video_path))[0]))
    
    control_dict = read_json(json_path)
    if control_dict['recalib_bool'] == True:
        txt2csv(calib_path, 'Error distribution in physical coordinate', 'RMS', csv_path)
    
    data_dict = {'grid_len_phys':5,
                 'grid_len_pix':32}

    calib = calibrator(grid_len_phys=float(data_dict['grid_len_phys']), grid_len_pix=int(data_dict['grid_len_pix']), csv_path=csv_path)
    
    if control_dict['recalib_bool'] == True:
        calib.pre_processing()
    
    image = cv2.imread(orignal_image_path, 0)
    image = calib.transpose(image)
    cv2.imwrite(calibed_image_path, image)
    
    output_size = (image.shape[1],image.shape[0])
    video_processing(orignal_video_path,calibed_video_path,calib.transpose,output_size)()
