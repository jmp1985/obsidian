'''
Image processing main class for testing and developing
.. moduleauthor:: Fiona Young
'''
import numpy as np
from obsidian.imp.processor import Processor
import matplotlib.pyplot as plt
import skimage.io as skio
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import *
import os.path
from obsidian.fex.trace import Trace
from obsidian.fex.powerSpect import PowerSpect
from obsidian.fex.extractor import FeatureExtractor as Fex
from glob import glob
import pickle
import gc

def fname(f):
  '''
  :param f: string, filepath
  '''
  return os.path.splitext(os.path.basename(f))[0]

def main1():
  
  ##################
  #   directories  #
  ##################
  
  # image data directory
  img_data_dir = input("Enter directory path for images: ")
  if img_data_dir == '':
    img_data_dir = 'data/realdata/npy_files/tray2/a5/grid'
  assert os.path.exists(img_data_dir), " not a real directory "

  # background data directory
  bg_data_dir = input("Enter directory path for background data: ")
  if bg_data_dir == '':
    bg_data_dir = 'data/realdata/npy_files/tray2/g1/grid'
  assert os.path.exists(bg_data_dir), " not a real directory "
  
  ######################
  # read in data files #
  ######################
  
  print("Loading data...")

  #coll = skio.ImageCollection(data_dir+'/*')
  coll = {fname(f) : np.load(f) for f in glob(img_data_dir+'/*.npy')[54:60]}

  # read header file to determine beam centre coordinates
  head = open(img_data_dir+'/header.txt', 'r')
  
  for l in head: # search lines
    if 'Beam_xy' in l:
      # extract (x, y) info from Beam_xy line
      beam_centre = eval(l[l.find('('):l.find(')')+1])
      # beam centre is stored in xy coordinates. transform into pixel coordinates (row, col)
      beam_centre = tuple(reversed(beam_centre))
  
  ############################
  # read in background data  #
  ############################
  print("Loading background...")
  background = np.load('obsidian/datadump/tray2_bg.npy')

  ##################
  #   processing   #
  ##################
  
  print("Pre-prossessing images...")

  # use subset of data
  data = coll

  # display unprocessed images
  imgDisp = ImgDisp(list(data.values()), background)
  fig1, ax1 = imgDisp.disp()
  fig1.suptitle('Preprocessed images')
  
  process = Processor(data, background)

  process.rm_artifacts(value=500)
  process.background()
  #fig0, ax0 = process.correct_and_filter()

  # display processed images
  newImgDisp = ImgDisp(list(process.processedData.values()), background)
  fig2, ax2 = newImgDisp.disp()
  fig2.suptitle('Processed images')
  
  # save
  #process.dump_save()
  
  ######################
  #  feature analysis  #
  ######################
  
  print("Extracting profiles...")

  fex = Fex(process.processedData)

  fex.meanTraces(centre=(1319.36, 1249.42), rmax=500, nangles=15)
  fex.dump_save()

  # extract mean trace from one image for display purposes
  demo_img = list(process.processedData.values())[4]
  angles = np.linspace(89,-89, 15)

  tr = Trace(demo_img.shape, angles, centre=(1319.36, 1249.42), rmax=500)

  traces, meanVals = tr.meanTrace(demo_img)
  fig3, ax3 = tr.display(demo_img, traces, meanVals)  
  
  # power spectrum of mean trace
  #powSpect = PowerSpect(profiles[-1]) # final entry is mean
  #fig4, ax4 = powSpect.display(powSpect.spect())
  
  plt.show()

def main2():
  
  ##################
  #   directories  #
  ##################
  '''
  # Image data directory
  img_data_dir = '/media/Elements/obsidian/diffraction_data/tray'+input("Complete directory path for images: /media/Elements/obsidian/diffraction_data/tray") 
  assert os.path.exists(img_data_dir), " not a real directory "

  # Tray number
  tray_nr = int(input("Enter tray number: "))
  assert tray_nr in (1, 2, 4, 5), "Available tray numbers are 1, 2, 4, 5"
  
  # Data batch id
  ID = input("Enter a batch ID to help identify pickle files saved to obsidian/obsidian/datadump: ")
  '''
  trays = {}
  
  done = False

  while not done:
    tray = input("Enter tray number (or press enter if done): ")
    assert tray in ('1', '2', '4', '5', '')
    wells = input("Enter comma separated well names for tray number {}: ".format(tray)).split(',')
    if tray == '' or wells == ['']:
      done = True
    else:
      trays[int(tray)] = wells

  for tray_nr in trays.keys():
 
    ############################
    # read in background data  #
    ############################
      
    print("Loading background data for tray {}...".format(tray_nr))
      
    background = np.load('obsidian/datadump/tray{}_background.npy'.format(tray_nr))
    
    for well in trays[tray_nr]:

      img_data_dir = '/media/Elements/obsidian/diffraction_data/tray{}/{}/grid'.format(tray_nr, well)
      assert os.path.exists(img_data_dir), "{} not found".format(img_data_dir)
      ID = 'T{}{}'.format(tray_nr, well)

      batched_files = split_data( glob(img_data_dir+'/*.npy'), 150 ) # batch size of 150
      print([len(l) for l in batched_files])

      batchIDs = ['{}-{}'.format(ID, i) for i in range(len(batched_files))]
      i = 0
      
      for files in batched_files: 
        
        batchID = batchIDs[i]
        print('Batch nr: ', i)
        
        ######################
        # read in data files #
        ######################
        
        print("Loading image data...") 

        data = {f : np.load(f) for f in files}
        names = list(data.keys())
        
        ##################
        #   processing   #
        ##################
        
        print("Pre-prossessing images...")
        
        process = Processor(data, background)

        process.rm_artifacts(value=500)
        process.background()
        
        ######################
        #  feature analysis  #
        ######################
        
        print("Extracting profiles...")

        fex = Fex(process.processedData)

        fex.meanTraces(centre=(1318.37, 1249.65), rmax=400, nangles=20)

        ####################
        #    saving        #
        ####################
        
        #print("Saving...")
        #process.dump_save(ID)

        print("Saving profiles to datadump/{}_profiles.pickle...".format(batchID))
        fex.dump_save(batchID)
        
        del data
        del fex
        del process
        i += 1

      ################
      # join batches #
      ################

      paths = ['obsidian/datadump/{}_profiles.pickle'.format(batchID) for batchID in batchIDs]
      join_files('obsidian/datadump/{}_profiles.pickle'.format(ID), paths)

if __name__ == '__main__':
  main2()

