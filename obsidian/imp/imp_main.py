'''
Image processing main class for testing and developing
.. automodule:: imp_main
.. moduleauthor:: Fiona Young
'''
import numpy as np
from skimage import data
from sbtr_bg import Sbtr_bg
import matplotlib.pyplot as plt
import skimage.io as skio
from obsidian.utils.imgdisp import ImgDisp
import os.path
from obsidian.fex.trace import Trace
from obsidian.fex.powerSpect import PowerSpect

def main():
  
  data_dir = input("Enter directory path for images")
  assert os.path.exists(data_dir), " not a real directory "
  # read in data files
  coll = skio.ImageCollection(data_dir+'/*.png')
  
  # (arbitrarily) choose background image
  background = coll[0]
  data = coll[1:]
  
  # display unprocessed images
  imgDisp = ImgDisp(data, background) # take first image as background image
  fig1, ax1 = imgDisp.disp()
  fig1.suptitle('Preprocessed images')

  # subtract background image from files 
  sbtrbg = Sbtr_bg(background)
  processedData = sbtrbg.subtract(data)

  # display processed images
  newImgDisp = ImgDisp(processedData, background)
  fig2, ax2 = newImgDisp.disp()
  fig2.suptitle('Images with background subtracted')
  
  # extract mean trace from one image
  tr = Trace(processedData[3])
  lines, profiles = tr.meanTrace((90, 0, 45, -30, 20, 70, 65, 10))
  fig3, ax3 = tr.display(lines, profiles)  
  
  # power spectrum of mean trace
  powSpect = PowerSpect(profiles[-1]) # final entry is mean
  fig4, ax4 = powSpect.display(powSpect.spect())
  plt.show()


main()
