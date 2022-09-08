from tqdm import tqdm
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import multiprocessing
from multiprocessing import Pool
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from experiment.LightGBM.func.stack_memmap import stack_memmap
import shutil

def velocity_gradient_tensor(a, b, eps1, eps2, e4, omega):
    #                                        [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]
    # returns a velocity gradient tensor D = [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]
    #                                        [ ∂w/∂x  ∂w/∂y  ∂w/∂z ]
    # having stretching rates of eps1, eps2 along unit vectors e1, e2 and vorticity omega*e4.
    # a, b: numpy array (shape = (3,)) to calculate unit orthogonal vevtors e1, e2, e3.
    #       a and b should not be parallel!
    # eps1, eps2: stretching rates along e1, e2.
    #             eps3, that along e3, becomes -eps1 - eps2 to satisfy
    #             an incpmpressible continuity equation.
    # omega: magnitude of vorticity having its direction of e4
    # e4: numpy array (shape = (3,)). Its length should be unity.
    #
    #                    [ ∂u/∂x  ∂u/∂y  ∂u/∂z ]
    # 速度こう配テンソルD = [ ∂v/∂x  ∂v/∂y  ∂v/∂z ]
    #                    [ ∂w/∂x  ∂w/∂y  ∂w/∂z ]
    # で，単位ベクトルe1, e2方向に伸び率eps1, eps2を持ち，渦度omega*e4を持っているものを返す．
    # a, b: numpy配列(shape = (3,))で，これらから単位直行ベクトルe1, e2, e3を求める．
    #       aとbは並行であってはいけない！
    # eps1, eps2: e1, e2方向の伸び率．
    #             e3方向の伸び率eps3は非圧縮性流体の連続の式を満たすべく-eps1 - eps2になる．
    # omega: e4と方向を持つ渦度の大きさ
    # e4: numpy配列(shape = (3,))で，長さは1でなければならない．

    e1 = a/np.linalg.norm(a, ord = 2)
    e2 = b - np.dot(e1, b)*e1
    e2 = e2/np.linalg.norm(e2, ord = 2)
    e3 = np.cross(e1, e2)
    e4 = e4/np.linalg.norm(e4, ord = 2)
    return (eps1*e1.reshape(-1, 1)*e1 + eps2*e2.reshape(-1, 1)*e2 - (eps1 + eps2)*e3.reshape(-1, 1)*e3 +
        0.5*omega*np.array([[0.0, -e4[2], e4[1]], [e4[2], 0.0, -e4[0]], [-e4[1], e4[0], 0.0]]))

def mk_API(eq_coef, sigma_l, xp, yp, zp, d_p, white_noise, inner_frame):
    x = np.arange(-inner_frame/2, inner_frame/2)
    y = np.arange(-inner_frame/2, inner_frame/2).reshape(-1,1)
    img = np.sum(eq_coef*np.exp(-zp**2/(sigma_l/2)**2)*np.exp(-((x-xp)**2+(y-yp)**2)/(d_p/2)**2),axis=0)+white_noise # 輝度の計算
    img[img >= 255] = 255 # キャスト時のオーバーフロー対策
    img = img.astype(np.uint8)
    return img

def data_gen():
    while True:
        a = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        b = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        if np.linalg.norm(a, ord = 2) >= 0.01 and np.linalg.norm(b, ord = 2) >= 0.01: # a, bのノルムが0でない
            theta = np.rad2deg(np.arccos(np.linalg.norm(np.dot(a, b))/(np.linalg.norm(a)*np.linalg.norm(b)))) # a, bの角度を算出
            if theta >= 1: # a, bが平行でない
                break
    eps_max = 0.025 # 伸び率の最大値
    eps1 = random.uniform(-math.sqrt(2/3)*eps_max, math.sqrt(2/3)*eps_max)
    eps2 = random.uniform(0.5*(-eps1-math.sqrt(-3*eps1**2+2*eps_max**2)), 0.5*(-eps1+math.sqrt(-3*eps1**2+2*eps_max**2)))
    while True:
        e4 = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        if np.linalg.norm(e4, ord=2) >= 0.01: # e4のノルムが0でない
            break
    omega_max = 0.05 # omegaの最大値
    omega = random.uniform(-omega_max, omega_max)
    tensor = velocity_gradient_tensor(a, b, eps1, eps2, e4, omega)
    outer_frame = 64
    inner_frame = 32
    particle_num = 120
    depth = 64
    sigma_l = 19.3 # z=+-8ピクセルの時に輝度128となるようなレーザー厚さ
    d_p = 3
    xp = np.random.uniform(-outer_frame/2, outer_frame/2, particle_num).reshape(-1, 1, 1)
    yp = np.random.uniform(-outer_frame/2, outer_frame/2, particle_num).reshape(-1, 1, 1)
    zp = np.random.uniform(-depth/2, depth/2, particle_num).reshape(-1, 1, 1)
    white_noise = random.uniform(0,255*0.01)
    eq_coef = 255
    
    img_0 = mk_API(eq_coef, sigma_l, xp, yp, zp, d_p, white_noise, inner_frame)
    
    main_u = random.uniform(-8, 8)
    main_v = random.uniform(-8, 8)
    main_w = random.uniform(-8, 8)
    u = main_u + tensor[0][0]*xp + tensor[0][1]*yp + tensor[0][2]*zp
    v = main_v + tensor[1][0]*xp + tensor[1][1]*yp + tensor[1][2]*zp
    w = main_w + tensor[2][0]*xp + tensor[2][1]*yp + tensor[2][2]*zp
    
    xp = xp + u
    yp = yp + v
    zp = zp + w
    img_1 = mk_API(eq_coef, sigma_l, xp, yp, zp, d_p, white_noise, inner_frame)
    
    return img_0, img_1, main_u, main_v, main_w

def main(args):
    process_id, save_dir, data_num = args
    process_dir = os.path.join(save_dir, str(process_id))
    if not os.path.exists(process_dir):
        os.mkdir(process_dir)
    
    input_dtype = np.float16
    img_size = 32
    label_dtype = np.float16
    input_mem_path = os.path.join(process_dir, 'input.npy')
    label_mem_path = os.path.join(process_dir, 'label.npy')
    input_memmap = np.memmap(input_mem_path, dtype=input_dtype, mode='w+', shape=(data_num, img_size, img_size, 2))
    label_memmap = np.memmap(label_mem_path, dtype=label_dtype, mode='w+', shape=(data_num, 3))
    for i in tqdm(range(data_num)):
        img_0, img_1, main_u, main_v, main_w = data_gen()

        data = np.array([img_0, img_1])
        data = data.transpose(1,2,0) # チャンネルラスト
        data = data.astype(input_dtype)/255 # 正規化
        input_memmap[i] = data
        
        label_memmap[i] = np.array([main_u, main_v, main_w]).astype(label_dtype)
    
if __name__ == '__main__':
    save_dir = input('input save dir > ')
    logical_processor = int(input('何コアで並列計算させますか？上限は{}です>'.format(multiprocessing.cpu_count())))
    total_data_num = int(input('input data num > '))
    args = [[process_id, save_dir, total_data_num//logical_processor] for process_id in range(logical_processor)]
    p = Pool(logical_processor)
    p.map(main,args)
    p.close()
    
    memmap_path_list = [os.path.join(save_dir, str(i), 'input.npy') for i in range(logical_processor)]
    stack_memmap_path = os.path.join(save_dir, 'input.npy')
    stack_memmap(memmap_path_list, stack_memmap_path, 32, 32, dtype=np.float16)
    memmap_path_list = [os.path.join(save_dir, str(i), 'label.npy') for i in range(logical_processor)]
    stack_memmap_path = os.path.join(save_dir, 'label.npy')
    stack_memmap(memmap_path_list, stack_memmap_path, 1, 1, dtype=np.float16)
    
    for dir_path in [os.path.join(save_dir, str(i)) for i in range(logical_processor)]:
        shutil.rmtree(dir_path)