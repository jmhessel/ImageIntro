import numpy as np

from PIL import Image

def isolate_channel(m_array, channel=0):
    '''
    Given a height, width, channel image,
    returns a numpy array of the same shape, except that
    all channels are zeroed except for the specified one.
    '''
    new_array = np.array(m_array)
    all_channels = m_array.shape[-1]
    for idx in range(all_channels):
        if idx == channel: continue
        new_array[:,:,idx] = 0
    return new_array

def compute_mean_color(image):
    '''
    returns the mean color of an rgb image
    '''
    if type(image) is str:
        image = np.array(Image.open(image))
    if not type(image) is np.array:
        image = np.array(image)
    mean_r = np.mean(image[:,:,0].flatten())
    mean_g = np.mean(image[:,:,1].flatten())
    mean_b = np.mean(image[:,:,2].flatten())
    return mean_r, mean_g, mean_b
    
def constant_color_image(r, g, b, size=100):
    '''
    given r, g, b values, returns a size x size image of only
    that color.
    '''
    r,g,b = np.round([r,g,b])
    m_arr = np.empty((size, size, 3), dtype=np.uint8)
    m_arr[:,:,0] = r
    m_arr[:,:,1] = g
    m_arr[:,:,2] = b
    return Image.fromarray(m_arr)
