import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.my_json import read_json, write_json


position = input('which camera is on? (side or bottom) >')
save_dir = input('input save dir >')

print('enter: 写真撮影＆保存')
print('q: プログラム終了')

# 動画ファイル保存用の設定
fps = 220                    # カメラのFPSを取得
w = 640              # カメラの横幅を取得
h = 480             # カメラの縦幅を取得
fourcc = cv2.VideoWriter_fourcc('Y', '8', '0', '0')        # 動画保存時のfourcc設定（mp4用）

if position == 'side':
    camera_order = 0
elif position == 'bottom':
    camera_order = 1
camera = cv2.VideoCapture(camera_order)                              # カメラCh.(ここでは0)を指定

# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
i = 0
while True:
    ret, frame = camera.read() # フレームを取得
    cv2.imshow('camera', frame) # フレームを画面に表示
    
    # エンターキーで画像保存
    if cv2.waitKey(1) == 13:
        cv2.imwrite(os.path.join(save_dir, '{}.bmp'.format(i)), frame)
        i += 1
    
    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# 撮影用オブジェクトとウィンドウの解放
camera.release()
cv2.destroyAllWindows()