#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 21:07 2020/12/16 

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import math

def fluid_func(cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3,cxy,cxz,cyz,x,y,z):
    # psi_x=ax^2+bxy+cxz+dy^2+eyz+fz^2+gx+hy+iz
    # psi_y=aa*x^2+bb*x*y+cc*x*z+dd*y^2+ee*y*z+ff*z^2+gg*x+hh*y+ii*z
    # psi_z=aaa*x^2+bbb*x*y+ccc*x*z+ddd*y^2+eee*y*z+fff*z^2+ggg*x+hhh*y+iii*z
    
    ux = cx1*z+cx2*y-cxz*x+cxy*x+cx3
    uy = cy1*z+cyz*y-cxy*y+cy2*x+cy3
    uz = -cyz*z+cxz*z+cz1*y+cz2*x+cz3
    
    return ux,uy,uz

def mk_image(image_num,eq_coef,sigma_l,uxy,u_random,cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3,cxy,cxz,cyz,theta,dir_name,fold_name,d_p_min,d_p_max,depth=32,height=64,width=64,size=32,times=2,particle_num=200):
    #depth:輝度計算時に考慮する深さ
    #height:輝度計算時に考慮する高さ
    #width:輝度計算時に考慮する幅
    #size:画像サイズ
    #times:輝度計算時に考慮するサイズ/画像サイズ
    
    x_p = np.random.uniform(0,width-1,particle_num).reshape(-1,1,1)
    y_p = np.random.uniform(0,height-1,particle_num).reshape(-1,1,1)
    z_p = np.random.uniform(0,depth-1,particle_num).reshape(-1,1,1)
    d_p = np.random.uniform(d_p_min,d_p_max,particle_num).reshape(-1,1,1)
    
    x = np.arange(width)
    y = np.arange(height).reshape(-1,1)
   
    white_noise = random.uniform(0,255*0.01)
    luminance_array = np.sum(eq_coef*np.exp(-(z_p - depth/2)**2/sigma_l**2)*np.exp(-((x-x_p)**2+(y-y_p)**2)/(d_p/2)**2),axis=0)+white_noise # 輝度の計算
    image = luminance_array[size*(times - 1)//2:size*(times + 1)//2, size*(times - 1)//2:size*(times + 1)//2]
    
    file_name = 'origin_{}_{}'.format(fold_name,image_num)
    
    cv2.imwrite(os.path.join(dir_name,fold_name,file_name + '.png'), image)
    
    
    
    ux = cx1*z_p+cx2*y_p-cxz*x_p+cxy*x_p+cx3
    uy = cy1*z_p+cyz*y_p-cxy*y_p+cy2*x_p+cy3
    uz = -cyz*z_p+cxz*z_p+cz1*y_p+cz2*x_p+cz3
    
    rotated_ux = ux*math.cos(theta)-uy*math.sin(theta)
    rotated_uy = uy*math.cos(theta)+ux*math.sin(theta)
    
    ux = rotated_ux/uxy*u_random
    uy = rotated_uy/uxy*u_random
    uz = uz/uxy*u_random
    
    x_p = x_p+ux
    y_p = y_p-uy
    z_p = z_p+uz
    
    white_noise = random.uniform(0,255*0.01)
    luminance_array = np.sum(eq_coef*np.exp(-(z_p - depth/2)**2/sigma_l**2)*np.exp(-((x-x_p)**2+(y-y_p)**2)/(d_p/2)**2),axis=0)+white_noise # 輝度の計算
    image = luminance_array[size*(times - 1)//2:size*(times + 1)//2, size*(times - 1)//2:size*(times + 1)//2]
    
    file_name = 'next_{}_{}'.format(fold_name,image_num)
    
    cv2.imwrite(os.path.join(dir_name,fold_name,file_name + '.png'), image)
    
    
def main(args):
    
    fold_name,logical_processor,hope_dataset_num = args
    
    #基本設定**********************************************************************************************
    eq_coef = 240
    sigma_l=5
    size = 32
    times = 2
    depth = 32
    width = size*times
    height = size*times
    max_uz = 0.5
    particle_num = 75
    d_p_min = 2.4
    d_p_max = 2.6
    
    #*****************************************************************************************************
    
    break_image_num = hope_dataset_num//logical_processor
    
    dir_name = 'result'
    fold_name = str(fold_name)
    data_output_path = os.path.join(dir_name,'dataset_output_{}.txt'.format(fold_name))
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    if not os.path.exists(os.path.join(dir_name,fold_name)):
        os.mkdir(os.path.join(dir_name,fold_name))
    
    with open (data_output_path,'w') as f_write:
        f_write.write('# ux   uy   uz \n')
    
    image_num = 0
    
    
    bar = tqdm(total = break_image_num,mininterval=1)
    bar.set_description('fold_name(logical_processor):{}'.format(fold_name))
    while True:
        
        cx1 = random.uniform(-0.1/5,0.1/5)
        cx2 = random.uniform(-0.1/32,0.1/32)
        cx3 = random.choice([-1,1])*random.uniform(7.3,7.3)
        cy1 = random.uniform(-0.1/5,0.1/5)
        cy2 = random.uniform(-0.1/32,0.1/32)
        cy3 = random.choice([-1,1])*random.uniform(7.3,7.3)
        cz1 = random.uniform(-0.01/32,0.01/32)
        cz2 = random.uniform(-0.01/32,0.01/32)
        cz3 = random.choice([-1,1])*random.uniform(0.433,0.433)
        cxy = random.uniform(-0.1/32,0.1/32)
        cxz = random.uniform(-0.001,0.001)
        cyz = random.uniform(-0.001,0.001)
        
        x = width//2
        y = height//2
        z = depth//2
        
        ux_center,uy_center,uz_center = fluid_func(cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3,cxy,cxz,cyz,x,y,z)
        
        theta = math.radians(random.uniform(-180,180))
        rotated_ux = ux_center*math.cos(theta)-uy_center*math.sin(theta)
        rotated_uy = uy_center*math.cos(theta)+ux_center*math.sin(theta)
        
        uxy = math.sqrt(rotated_ux**2+rotated_uy**2)
        
        u_random = random.uniform(0,8)
        
        rotated_ux = rotated_ux/uxy*u_random
        rotated_uy = rotated_uy/uxy*u_random
        uz_center = uz_center/uxy*u_random
        
        # 長さ調整後のzの速度が0.5であるか
        gridwidth=1
        x, y = np.meshgrid(np.arange(0, width, gridwidth), np.arange(0, height,gridwidth))
        z = depth//2
        ux,uy,uz = [i/uxy*u_random for i in fluid_func(cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3,cxy,cxz,cyz,x,y,z)]
        if np.all(abs(uz) <= max_uz):
            
            data_output = np.array([rotated_ux,rotated_uy,uz_center]).reshape(1, -1)
            with open(data_output_path,'a') as f_write:
                np.savetxt(f_write,data_output)
            
            
            mk_image(image_num,eq_coef,sigma_l,uxy,u_random,cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3,cxy,cxz,cyz,theta,dir_name,fold_name,d_p_min,d_p_max,depth=depth,height=height,width=width,size=size,times=times,particle_num=particle_num)
            
            
            # LX, LY=32,32
            # gridwidth=8
            # x, y = np.meshgrid(np.arange(0, 3*LX, gridwidth), np.arange(0, 3*LY,gridwidth))
            # z = 16
            # ux,uy,uz = [i/uxy*u_random for i in fluid_func(cx1,cx2,cx3,cy1,cy2,cy3,cz1,cz2,cz3,cxy,cxz,cyz,x,y,z)]
            # graph_ux = ux*math.cos(theta)-uy*math.sin(theta)
            # graph_uy = uy*math.cos(theta)+ux*math.sin(theta)
            # plt.figure()
            # plt.quiver(x,y,graph_ux,graph_uy,color='red',angles='xy',scale_units='xy', scale=1)
            # plt.xlim([0,3*LX])
            # plt.ylim([0,3*LY])
            # plt.xticks(np.arange(0,96,32))
            # plt.yticks(np.arange(0,96,32))
            # plt.grid()
            # plt.draw()
            # fig_path = os.path.join(dir_name,fold_name,'fig_0_{}'.format(image_num))
            # plt.savefig(fig_path)
            
            
            image_num += 1
            bar.update(1)
            # print(image_num)
            
            if image_num >= break_image_num:
                break

if __name__ =='__main__':
    os.chdir(os.path.dirname(__file__))
    logical_processor = int(input('何コアで並列計算させますか？上限は{}です>'.format(multiprocessing.cpu_count())))
    hope_dataset_num = int(input('データセット数>'))
    args = [[fold_name,logical_processor,hope_dataset_num] for fold_name in range(logical_processor)]
    p = Pool(logical_processor)
    p.map(main,args)
    
    
    result_path = os.path.join(os.path.dirname(__file__),'result')
    
    
    
    #基本設定*******************************************************************
    stick_output_list = [os.path.join(result_path,'dataset_output_{}.txt'.format(i)) for i in range(logical_processor)]
    #**************************************************************************
    
    print(stick_output_list)
    
    with open(os.path.join(result_path,'dataset_output.txt'),'w') as f:
        f.write('# ux   uy   uz \n')
    
    for i in tqdm(stick_output_list):
        with open(os.path.join(result_path,'dataset_output.txt'),'a') as f:
            np.savetxt(f,np.loadtxt(i))
    
