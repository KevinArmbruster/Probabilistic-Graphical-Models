# Author Marcus Klasson (mklas@kth.se)
# PGM tutorial on MRFs
# Loopy belief propagation in image denoising

import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

def save_np_array_as_png(fname, array):
    plt.imsave('%s.png' %(fname), array, cmap='gray')

def load_image_as_np_array(fname):
    with Image.open(fname).convert('L') as img:
        img_resized = img.resize((128, 96))
    return np.asarray(img_resized)

def save_image_as_np_array(fname, new_fname='img', height=128, width=96):
    with Image.open(fname).convert('L') as img:
        img = img.resize((height, width))
    img = np.asarray(img) 
    np.save(new_fname, img)