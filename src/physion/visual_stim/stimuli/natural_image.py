import os, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image
from physion.visual_stim.preprocess_NI import load,\
        img_after_hist_normalization, hist_match_to_average,\
        adapt_to_screen_resolution, list_NI_files

#######################################
##  ----    NATURAL IMAGES    --- #####
#######################################

params = {"Image-ID":1,
          "contrast":0.3,
          "x-center":0,
          "y-center":0,
          "radius":30
}


NORMALIZATIONS = ['equalization', 'histogram-matching', 'none']

def get_NaturalImages_as_array(screen, image_IDs=None,
                               normalization='equalization'):
    """
    returns a dictionary {Image-ID: normalized image}

    only the images in "image_IDs" are loaded (all images of the bank if None)
        "Image-ID" = n  corresponds to the file named "n+1" (see list_NI_files)

    normalization (set by the "NI-normalization" key of the protocol):
        'equalization'       -> uniform brightness distribution for each image (default)
        'histogram-matching' -> each image matched to the average histogram
                                    of the images loaded (i.e. of the protocol)
        'none'               -> images as loaded and resized
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError('NI-normalization "%s" not recognized, should be one of: %s' %\
                         (normalization, ', '.join(NORMALIZATIONS)))

    NI_FOLDER = os.path.join(str(pathlib.Path(__file__).resolve().parents[1]), 'NI_bank')

    if not os.path.isdir(NI_FOLDER):
        print(' [!!]  Natural Images folder not found !!! [!!]  ')
        IDs = range(5) if image_IDs is None else image_IDs
        return {int(i):np.ones((10,10))*0.5 for i in IDs}

    files = list_NI_files(NI_FOLDER)
    if image_IDs is None:
        image_IDs = range(len(files))

    NIarray = {}
    for ID in np.unique(np.array(image_IDs, dtype=int)):
        if not (0 <= ID < len(files)):
            raise IndexError('Image-ID %i out of range: NI_bank has %i images (Image-ID 0 to %i)' %\
                             (ID, len(files), len(files)-1))
        img = load(os.path.join(NI_FOLDER, files[ID])).T
        new_img = np.rot90(adapt_to_screen_resolution(img, screen), k=3)
        if normalization=='equalization':
            NIarray[int(ID)] = img_after_hist_normalization(new_img)
        else:
            NIarray[int(ID)] = new_img

    if normalization=='histogram-matching':
        # needs all images -> done once they are all loaded
        NIarray = hist_match_to_average(NIarray, verbose=True)

    return NIarray

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        # initializing set of NI
        # (only the images used in the protocol)
        self.NIarray = get_NaturalImages_as_array(self.screen,
                                        image_IDs=self.experiment['Image-ID'],
                                        normalization=protocol.get('NI-normalization',
                                                                   'equalization'))

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):

        im0 = np.rot90(\
                self.NIarray[int(self.experiment['Image-ID'][index])], 
                        k=1)
        im1 = self.experiment['bg-color'][index]+\
            self.experiment['contrast'][index]*(im0-0.5)
        
        return self.blank_surround(im1, 
                        bg_color=self.experiment['bg-color'][index],
                          xcenter=self.experiment['x-center'][index],
                          zcenter=self.experiment['y-center'][index],
                          radius = self.experiment['radius'][index])
    
"""
    def plot_stim_picture(self, episode, parent=None, 
                          vse=True, ax=None, label=None,
                          time_from_episode_start=0):

        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1)

        img = ax.imshow(\
            self.image_to_frame(\
                self.get_image(episode,
			                   time_from_episode_start=time_from_episode_start),
                               psychopy_to_numpy=True),
                        cmap='gray', vmin=0, vmax=1,
                            origin='lower',
                            aspect='equal')

        ax.axis('off')

        return ax
"""

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('natural-image')
    print(params)

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
