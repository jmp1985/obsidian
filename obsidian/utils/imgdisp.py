'''
Tool for displaying a set of images side by side
'''

import matplotlib.pyplot as plt
import numpy as np
from skimage import data 
from skimage.draw import circle_perimeter_aa
from random import randint
from skimage import io as skio
import math

class ImgDisp():
  '''
  Display a collection of images in a compact arrangement
  '''
  def __init__(self, collection, bgimage=None):
    '''
    :param collection: collection of images to be displayed
    :param bgimage: background image to be displayed alongside
    '''
    self.collection = collection
    self.bgimage = bgimage
    self.nimages = len(collection)
    
    # add one for background image
    if np.all(self.bgimage != None):
      self.nimages += 1

    self.shape = (int(math.sqrt(self.nimages)),
                  math.ceil(self.nimages/(int(math.sqrt(self.nimages)))))

  def disp(self):
    '''
    Display image collection together with background image if specified

    :returns: fig, axes
    '''
    fig, axes = plt.subplots(self.shape[0], self.shape[1])
    
    for i in range(len(self.collection)):
      axes.flat[i].imshow(self.collection[i], cmap='binary',
                          interpolation='nearest', vmin=-1,
                          vmax=300)
    
    if np.all(self.bgimage != None):
      axes.flat[-1].imshow(self.bgimage, cmap='binary', 
                            interpolation='nearest', vmin=-1,
                            vmax=300)
      axes.flat[-1].set_title('Background image')
    
    for ax in axes.flatten():
      ax.axis('off')
    
    return fig, axes
