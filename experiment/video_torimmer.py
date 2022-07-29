import cv2
import numpy as np
from tqdm import tqdm

memmap_path = input('input memmap path >')
save_path = input('input save path >')
cut_end = int(input('input cut end >'))
cut_start = int(input('input cut start >'))
fps = 120
width = 640
height = 240

fourcc = cv2.VideoWriter_fourcc('Y','8','0','0')
video = cv2.VideoWriter(save_path,fourcc,fps,(width, height))
arr = np.memmap(memmap_path, dtype='uint8', mode='r').reshape(-1, height, width)
for frame in tqdm(arr[cut_start:cut_end]):
    frame = np.repeat(frame, 3).reshape(height,width,3).astype(np.uint8)
    video.write(frame)
video.release()