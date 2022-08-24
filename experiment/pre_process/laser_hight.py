import cv2
from matplotlib import pyplot as plt
import numpy as np

def binary_processing(img, threshold):
    ret, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

def extract_largest_part(binary_img):
    # 輪郭抽出
    contours = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 一番面積が大きい輪郭を選択する。
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    # 黒い画像に一番大きい輪郭だけ塗りつぶして描画する。
    out = np.zeros_like(binary_img)
    mask = cv2.drawContours(out, [max_cnt], -1, color=255, thickness=-1)
    # 背景画像と前景画像を合成
    img_EdgeDtection = np.where(mask==255, binary_img, out)
    return img_EdgeDtection

def calc_laser_height(img, fig_bool=False, fig_path=None, back_img=None):
    x = np.where(img==255)[1]
    y = np.where(img==255)[0]
    
    a, b = np.polyfit(x,y,1)
    
    if fig_bool:
        x = np.arange(0, img.shape[1])
        y = a*x+b
        fig = plt.figure()
        plt.plot(x, y, color='red')
        plt.title('height of laser (estimated by least squares)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
        img_rgb = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb, alpha=1)
        fig.savefig(fig_path)
    
    return a, b
    
if __name__ == '__main__':
    # 画像を読み込む。
    img_path = input('input img path > ')
    img_origin = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 二値化(閾値100を超えた画素を255にする。)
    threshold = 100
    binary_img = binary_processing(img_origin, threshold)
    cv2.imwrite('out_0.bmp', binary_img)
    
    # エッジ検出->大きい塊だけ取り出す
    img_EdgeDtection = extract_largest_part(binary_img)
    cv2.imwrite('out_1.png', img_EdgeDtection)
    
    # 最小二乗法でレーザーの高さを求める
    fig_path = 'fig.png'
    calc_laser_height(img_EdgeDtection, fig_bool=True, fig_path=fig_path, back_img=img_origin)