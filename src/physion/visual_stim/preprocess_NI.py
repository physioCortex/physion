import os, sys
import numpy as np
from scipy.interpolate import RectBivariateSpline

import itertools, string, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from skimage import color, io
    from skimage.util import img_as_float
except ModuleNotFoundError:
    print('"skimage" module not found')


def list_NI_files(folder):
    """
    image files of the NI bank, sorted by the number in their name
        (1.jpg, 2.jpg, ..., 10.png) and not alphabetically (1, 10, 2, ...)
        -> "Image-ID" = n  corresponds to the file named "n+1"
    """
    files = [f for f in os.listdir(folder)\
             if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    return sorted(files, key=lambda f: int(os.path.splitext(f)[0]))

def load(image_path):

    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = color.rgba2rgb(img) # drop alpha channel (e.g. PNG)
    if img.ndim == 3:
        img_bw = color.rgb2gray(img) # float in [0,1]
    else:
        img_bw = img_as_float(img) # grayscale: rescale uint8/uint16 to [0,1]
    print('Loaded image: ', image_path, ' with shape: ', img_bw.shape)
    
    return np.rot90(np.array(img_bw).T, k=1) # needs transpose + rotation
    
def img_after_hist_normalization(img, verbose=False):
    """
    for NATURAL IMAGES:
    histogram normalization to get comparable images
    """
    if verbose:
        print('Performing histogram normalization [...]')

    # expects img in [0,1] (negative values would silently wrap around in norm_cs)
    if img.min() < 0 or img.max() > 1:
        raise ValueError('img_after_hist_normalization expects values in [0,1], '
                         'got [%.3f, %.3f]' % (img.min(), img.max()))
    flat = np.array(1000*img.flatten(), dtype=int)

    cumsum = np.cumsum(np.histogram(flat, bins=np.arange(1001))[0])

    norm_cs = np.concatenate([(cumsum-cumsum.min())/(cumsum.max()-cumsum.min())*1000, [1]])
    new_img = np.array([norm_cs[f]/1000. for f in flat])

    return new_img.reshape(img.shape)


def adapt_to_screen_resolution(img, new_screen, verbose=False):

    if verbose:
        print('Adapting image to chosen screen resolution [...]')
    
    old_X = np.arange(img.shape[0])
    old_Y = np.arange(img.shape[1])
    
    new_X = np.linspace(0, img.shape[0], new_screen['resolution'][0])
    new_Y = np.linspace(0, img.shape[1], new_screen['resolution'][1])

    new_img = np.zeros(new_screen['resolution'])
    spline_approx = RectBivariateSpline(old_X, old_Y, img, 
                                        kx=1, ky=1)
    
    return spline_approx(new_X, new_Y)

    
if __name__=='__main__':

    from datavyz import ge
    NI_directory = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')
    
    image_number = 0
    filename = os.listdir(NI_directory)[image_number]
    img = load(os.path.join(NI_directory, filename))

    SCREEN = {'width':20, 'height':12, 'resolution':(1200, 800)}
    rescaled_img = adapt_to_screen_resolution(img, SCREEN)
    rescaled_img = img_after_hist_normalization(rescaled_img).T
    print(rescaled_img.shape)
    ge.image(rescaled_img)
    ge.show()
