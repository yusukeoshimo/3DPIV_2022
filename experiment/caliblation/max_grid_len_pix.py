import pandas as pd
import numpy as np
import os
import pandas as pd
import re
import json
import glob

def calc_grid_len_pix(csv_path):
    # girdの最大長さを計算
    origin_list = pd.read_csv(csv_path).loc[:,['grid_step_x', 'grid_step_y']].values.tolist()
    grid_len_pix_x = []
    grid_len_pix_y = []
    df = pd.read_csv(csv_path)
    for origin_x, origin_y in origin_list:
    
        id_origin = df.index[((df['grid_step_x'] == origin_x)&(df['grid_step_y'] == origin_y))].tolist()[0]
        id_next_x= df.index[((df['grid_step_x'] == origin_x+1)&(df['grid_step_y'] == origin_y))].tolist()
        if len(id_next_x) != 0:
            id_next_x = id_next_x[0]
            origin_pix_x = df.at[id_origin,'Given Xpix']
            origin_pix_y = df.at[id_origin,'Given Ypix']
            next_pix_x = df.at[id_next_x, 'Given Xpix']
            next_pix_y = df.at[id_next_x, 'Given Ypix']
            grid_len_pix_x.append(abs(origin_pix_x-next_pix_x))
        
        
        id_next_y= df.index[((df['grid_step_x'] == origin_x)&(df['grid_step_y'] == origin_y+1))].tolist()
        if len(id_next_y) != 0:
            id_next_y = id_next_y[0]
            origin_pix_y = df.at[id_origin,'Given Ypix']
            next_pix_y = df.at[id_next_y, 'Given Ypix']
            grid_len_pix_y.append(abs(origin_pix_y-next_pix_y))
        
        grid_len_pix = grid_len_pix_x + grid_len_pix_y
    print(max(grid_len_pix))

if __name__ == '__main__':
    csv_path = input('input path of grid.csv >')
    calc_grid_len_pix(csv_path)