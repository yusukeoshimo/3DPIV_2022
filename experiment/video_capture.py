import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.my_json import read_json, write_json


project_dir_path = input('input project dir path >')
json_path = os.path.join(project_dir_path, 'system', 'control_dict.json')

# 動画ファイル保存用の設定
fps = 220                    # カメラのFPSを取得
w = 640              # カメラの横幅を取得
h = 480             # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('Y', '8', '0', '0')        # 動画保存時のfourcc設定（mp4用）

camera0 = cv2.VideoCapture(0)                              # カメラCh.(ここでは0)を指定
video_path_side = os.path.join(project_dir_path, 'side', 'raw_video', '30120025.avi')
video = cv2.VideoWriter(video_path_side, fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

camera1 = cv2.VideoCapture(1)
video_path_bottom = os.path.join(project_dir_path, 'bottom', 'raw_video', '30120023.avi')
video1 = cv2.VideoWriter(video_path_bottom, fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

press_enter = input('Press enter, and video will be caputerd >')

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
while True:
    ret0, frame0 = camera0.read() # フレームを取得
    if (ret0 == True ) :
        cv2.imshow('camera0', frame0) # フレームを画面に表示　　　　　 
        video.write(frame0) # 動画を1フレームずつ保存する

    ret1, frame1 = camera1.read() # フレームを取得
    if (ret1 == True ) :
        cv2.imshow('camera1', frame1) # フレームを画面に表示
        video1.write(frame1) # 動画を1フレームずつ保存する
    
    # エンターキーでwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
camera0.release()
camera1.release()
cv2.destroyAllWindows()

control_dict = read_json(json_path)
control_dict['side']['raw_video_path'] = video_path_side
control_dict['bottom']['raw_video_path'] = video_path_bottom
write_json(json_path, control_dict)
