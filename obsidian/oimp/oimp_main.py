'''
Image processing main class for testing and developing
'''
import numpy as np
from obsidian.oimp.processor import Processor
import matplotlib.pyplot as plt
import skimage.io as skio
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, pickle_put, join_files, split_data, read_header
import os.path
from obsidian.fex.trace import Trace
from obsidian.fex.extractor import FeatureExtractor as Fex, radius_from_res
from glob import glob
import pickle
import gc

def fname(f):
  '''
  :param f: string, filepath
  '''
  return os.path.splitext(os.path.basename(f))[0]

def main1():
  '''
  '''
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
  '''
  '''
  ##################
  #   directories  #
  ##################

  trays = {}
  
  done = False

  while not done:
    tray = input("Enter tray number (or press enter if done): ")
    
    if tray == '':
      done = True
    else:
      wells = input("Enter comma separated well names for tray number {}: ".format(tray)).split(',')
      trays[int(tray)] = wells

  for tray_nr in trays.keys():
 
    ############################
    # read in background data  #
    ############################
      
    #print("Loading background data for tray {}...".format(tray_nr))
      
    #background = np.load('obsidian/datadump/tray{}_background.npy'.format(tray_nr))
    
    for well in trays[tray_nr]:

      img_data_dir = '/media/Elements/obsidian/diffraction_data/180726/tray{}/{}'.format(tray_nr, well)
      assert os.path.exists(img_data_dir), "{} not found".format(img_data_dir)
      ID = 'T{}{}'.format(tray_nr, well)

      batched_files = split_data( glob(img_data_dir+'/*.npy'), 150 ) # batch size of 150
      print([len(l) for l in batched_files])

      batchIDs = ['{}-{}'.format(ID, i) for i in range(len(batched_files))]
      i = 0
      
      max_res=7 #Angstrom

      header = os.path.join(img_data_dir, 'header.txt')
      params = ['Wavelength','Detector_distance','Pixel_size']
      info = read_header(header, params)
      wl = float(info['Wavelength'][0])
      L = float(info['Detector_distance'][0])
      pixel_size = float(info['Pixel_size'][0])
      rmax = radius_from_res(wl, max_res, L, pixel_size)

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
        
        process = Processor(data)

        process.rm_artifacts(value=500)
        #process.background(background)
        data = process.processedData
        ######################
        #  feature analysis  #
        ######################
        
        print("Extracting profiles...")

        fex = Fex(data)

        fex.meanTraces(centre=(1318.37, 1249.65), rmax=rmax, nangles=20)

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

