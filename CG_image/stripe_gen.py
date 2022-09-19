import os
from turtle import width
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def stripe_gen(width, height, a_low=0.0, a_hight=0.4, b_low=0.6):
    # 軸の回転
    dst_x = np.arange(width)
    dst_y = np.arange(height).reshape(-1, 1)
    phi = np.random.uniform(0, 2*np.pi)
    src_x = np.cos(phi)*dst_x - np.sin(phi)*dst_y
    src_y = np.sin(phi)*dst_x + np.cos(phi)*dst_y
    # x成分のサイン波の生成
    alpha_x = np.random.uniform(low=0, high=2*np.pi)
    frequency_x = np.random.uniform(low=0, high=2)
    wave_length_x = width/frequency_x
    intensity_x = np.sin((2*np.pi*src_x)/wave_length_x + alpha_x)
    # y成分のサイン波の生成
    alpha_y = np.random.uniform(low=0, high=2*np.pi)
    frequency_y = np.random.uniform(low=0, high=2)
    wave_length_y = height/frequency_y
    intensity_y = np.sin((2*np.pi*src_y)/wave_length_y + alpha_y)
    # 平均輝度分布を正規化
    intensity_x = 0.5*intensity_x + 0.5
    intensity_y = 0.5*intensity_y + 0.5
    # 二次元グラフにする
    intensity = intensity_x*intensity_y
    # 平均輝度分布の値域を1~0.6に
    a = np.random.uniform(low=a_low, high=a_hight)
    b = np.random.uniform(low=b_low, high=(1-a))
    intensity = a*intensity + b
    return intensity

if __name__ == '__main__':
    for i in range(10):
        intensity = stripe_gen(32, 32, a_low=0, a_hight=1, b_low=0)
        intensity = (30*intensity).astype(np.uint8)
        
        cv2.imshow('intensity', intensity)
        cv2.waitKey(1000)