'''
For extracting and handling intensity profiles.
'''

import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

class Trace():
  '''Trace extraction and manipulation class.

  :ivar int w: image width in pixels
  :ivar int h: image height in pixels
  :ivar tuple cent: Beam centre (either provided or assumed to be image centre)
  :ivar float rmax: maximum radius in pixels 
  :ivar list lines: array of line coordinates along which image values will be extracted
  '''
  def __init__(self, image_shape, angles, centre=None, rmax=None):
    '''Instantiaing initialises array of line coordinates corresponding to `angles`,
    and sets instance variables width, height, rmax and lines.

    :param tuple image_shape: image dimensions in pixels
    :param list angles: collection of angles in range (-90, 90)deg
    :param tuple centre: (row, col) pixel coordinates of beam centre. If not provided image assumed to be square and centred
    :param int rmax: radius in pixels of data to be extracted, corresponding to maximum resolution of interest
    '''
    # Set image dimensions
    self.w = image_shape[1] # image width is number of columns
    self.h = image_shape[0] # image height is number of rows
    
    # Set image / beam centre
    if centre is None:
      #assert(image_shape[0]==image_shape[1]), "image must be square if centre coordinates not specified"
      self.cent = (self.h/2, self.w/2)
    else:
      self.cent = centre

    # Set maximum radius
    if rmax is None or rmax > self.w or rmax > self.h:
      self.rmax = min(*self.cent, self.w-self.cent[1], self.h-self.cent[0])
    else:
      self.rmax = rmax

    # Set list of line coordinates
    self.lines = [ self.line(angle) for angle in angles ]

  def line(self, angle):
    '''Calculate coordinates of a single line through image centre, with length 2 x rmax.

    :param angle: angle w.r.t. horizontal axis in range (-90, 90] deg
    :returns: line coordinates
    '''
    assert(angle > -90 and angle <= 90),  "angle must be in range (-90, 90]"
    
    x0 = self.rmax*math.cos(math.radians(angle))
    y0 = self.rmax*math.sin(math.radians(angle))
    r0, c0 = self.cent[0]+y0, self.cent[1]-x0 # start row and column in pixel coordinates
    r1, c1 = self.cent[0]-y0, self.cent[1]+x0 # end row and column in pixel coordinates

    num = 2000 # number of points on line
    r, c = np.linspace(r0, r1, num), np.linspace(c0, c1, num)

    return np.vstack((r, c))

  def read_trace(self, line, img):
    '''Extract image values along a single line.

    :param np.array line: line coordinates as 2d np array
    :param img: input image
    :returns: image values along line
    '''

    return img[line[0,:].astype(np.int), line[1,:].astype(np.int)]
  
  def mean_trace(self, img):
    '''Calculate the mean image profile across all angles.
    
    :param img: input image
    :returns: individual traces and mean trace values 
    '''

    traces = [self.read_trace(line, img) for line in self.lines]
    meanVals = np.mean(traces, axis = 0)

    return traces, meanVals

  def display(self, img, traces, meanVals):
    '''Display image with lines overlaid and resulting trace plots.
    
    :param img: input image as numpy array
    :param list traces: individual traces from each line
    :param list meanVals: values averaged across all lines 
    :return: pyplot figure and axes objects
    '''
    # Set up figure gridspace and axes
    fig = plt.figure(dpi=300)
    gs = GridSpec(len(traces),2)
    axes = [plt.subplot(gs[0:-4,0]), plt.subplot(gs[-4:,0])] \
            + [plt.subplot(gs[i,1]) for i in range(len(traces))]
    
    # Show figure in first axis
    axes[0].imshow(img, cmap='binary', interpolation='nearest', vmin=-1, vmax=200)
    axes[0].set_axis_off() 
    
    # Overlay figure with lines indicating profiles
    for line in self.lines:
      print(line)
      axes[0].plot(line[1], line[0], linewidth=0.3, color='#5e9ec4')
    
    # Plot mean trace below figure
    axes[1].plot(meanVals, linewidth=0.2, color='#5e9ec4')
    axes[1].set_axis_off()
    
    # Plot individual trace values
    for i in range(len(traces)):
      axes[i+2].plot(traces[i], linewidth=0.2, color='#5e9ec4')
    
    return fig, axes
