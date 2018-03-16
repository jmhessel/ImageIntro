'''
Simple interface to convolution. 
It's not perfect, but this is the best I could do after tweaking a lot.
'''

from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage import exposure

def convolve(image, kernel):
    image = np.array(image)
    image, kernel = image.astype(np.float), kernel.astype(np.float)
    image /= np.max(image)
    res = convolve2d(image, kernel, 'same')
    res = exposure.equalize_adapthist(res/np.max(np.abs(res)), clip_limit=0.03)
    res *= 255
    res = res.astype(np.uint8)
    return Image.fromarray(res)

def equalize_exposure(image):
    res = np.array(image).astype(np.float)
    if np.min(res.flatten()) < 0:
        res += np.min(res.flatten())
    res /= np.max(res)
    res = exposure.equalize_adapthist(res/np.max(np.abs(res)), clip_limit=0.03)
    res *= 255
    res = res.astype(np.uint8)
    return Image.fromarray(res)
