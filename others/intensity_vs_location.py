import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

def main(img):
    fig, ax = plt.subplots()
    ax.set_title('intensity vs location')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    c = plt.imshow(img, cmap=plt.cm.jet)
    plt.colorbar()
    plt.clim(0, 255)
    plt.show()

if __name__ == '__main__':
    img_path = input('input img >')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    main(img)