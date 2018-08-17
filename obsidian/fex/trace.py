'''
Tool for extracting and handling intensity profiles
'''

import math
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from time import time
class Trace():
  '''
  Trace extraction and manipulation class

  :ivar int w: image width in pixels
  :ivar int h: image height in pixels
  :ivar list lines: array of line coordinates along which image values will be extracted
  '''
  def __init__(self, image_shape, angles, centre=None, rmax=None):
    '''
    Initialise array of line coordinates corresponding to `angles`. Set instance variables width, height, rmax and lines

    :param tuple image_shape: image dimensions in pixels    
    :param tuple centre: (row, col) pixel coordinates of beam centre. If not provided image assumed to be square and centred
    :param int rmax: radius in pixels of data to be extracted, corresponding to maximum resolution of interest
    '''
    self.w = image_shape[1] # image width is number of columns
    self.h = image_shape[0] # image height is number of rows
    
    if centre is None:
      #assert(image_shape[0]==image_shape[1]), "image must be square if centre coordinates not specified"
      self.cent = (self.h/2, self.w/2)
    else:
      self.cent = centre
      
    if rmax is None or rmax > self.w or rmax > self.h:
      self.rmax = min(*self.cent, self.w-self.cent[1], self.h-self.cent[0])
      print("Warning: rmax either not specified or exeeded image dimensions. ramx set to half maximum image dimension.")
    else:
      self.rmax = rmax
    
    self.lines = [ self.line(angle) for angle in angles ]

  def line(self, angle):
    '''
    Calculate coordinates of a single line through image

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
    # return line coordinates 
    return np.vstack((r, c))

  def readTrace(self, line, img):
    '''
    Extract image values along a single line

    :param np.array line: line coordinates as 2d np array
    :param img: input image
    :returns: line coordinates and image values along line
    '''
    # extract image values along line using bilinear interpolation
    #vals = map_coordinates(img, line, order=2)
    vals = img[line[0,:].astype(np.int), line[1,:].astype(np.int)]
    return vals
  
  def meanTrace(self, img):
    '''
    Calculate the mean image profile across all angles
    
    :param img: input image
    :returns: individual traces and mean trace values 
    '''
    #time this

    traces = [self.readTrace(line, img) for line in self.lines]
    meanVals = np.mean(traces, axis = 0)
    return traces, meanVals

  def display(self, img, traces, meanVals):
    '''
    Display image with lines overlaid and resulting trace plots
    
    :param img: input image
    :param list traces: individual traces from each line
    :param list meanVals: values averaged across all lines 
    :return: pyplot figure and axes objects
    '''
    # set up figure gridspace and axes
    fig = plt.figure(dpi=300)
    gs = GridSpec(len(traces),2)
    axes = [plt.subplot(gs[0:-4,0]),
    plt.subplot(gs[-4:,0])]+[plt.subplot(gs[i,1]) for i in range(len(traces))]
    
    # show figure in first axis
    axes[0].imshow(img, cmap='binary', interpolation='nearest', vmin=-1, vmax=200)
    axes[0].set_axis_off() 
    # overlay figure with lines indicating profiles
    for line in self.lines:
      print(line)
      axes[0].plot(line[1], line[0], linewidth=0.3, color='#5e9ec4')
    
    # plot mean trace below figure
    axes[1].plot(meanVals, linewidth=0.2, color='#5e9ec4')
    axes[1].set_axis_off()
    # plot trace values
    for i in range(len(traces)):
      axes[i+2].plot(traces[i])
    
    return fig, axes
