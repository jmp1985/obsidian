'''
Image processing main class for testing and developing
This module brings together the image processing and line extracion functions of obsidian. 
The directories containing files to be processed are taken
from user input (exactly how differs between main1 and main2) and the 
files are then passed through Processor and FeatureExtractor objects respectively, and the resulting
data dictionaries of form {imagepath:mean_profile_data} are saved to pickle files named with IDs
'''
import os, os.path
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, pickle_put, join_files, split_data, read_header
from obsidian.utils.bg_from_blanks import build_bg_files
from obsidian.fex.trace import Trace
from obsidian.fex.extractor import FeatureExtractor as Fex, radius_from_res
from obsidian.oimp.processor import Processor

def fname(f):
  '''
  :param f: string, filepath
  '''
  return os.path.splitext(os.path.basename(f))[0]

def get_img_dirs(root):
  '''
  Walk through root directory compiling a list of all bottom level directories (that is directories that contain only
  files and no subdirectories)
  '''
  bottom_dirs = {}
  for dirname, subdirList, fileList in os.walk(root, topdown=False):
    if len(subdirList)==0:
      ID = ''.join(dirname.split(os.sep)[-3:])
      tray = ''.join(dirname.split(os.sep)[-4:-2])
      bottom_dirs[dirname] = {'ID':ID, 'tray':tray}
      
  return bottom_dirs

def pipe(top, dump_path, max_res):
  '''
  This version takes a single top level directory as input and processes all nested files.
  IDs are derived from directory paths. Background files are created if not already existing
  '''
  ##################
  #   directories  #
  ##################
  

  bottom_dirs = get_img_dirs(top)
  build_bg_files(top, dump_path)

  for img_data_dir in bottom_dirs.keys():
  ############################
  # read in background data  #
  ############################
     
    assert os.path.exists(img_data_dir), "{} not found".format(img_data_dir)
    ID = bottom_dirs[img_data_dir]['ID']
    tray = bottom_dirs[img_data_dir]['tray']
    print("Working on {}...".format(ID))  
    
    if os.path.exists(os.path.join(dump_path, '{}_profiles.pickle'.format(ID))):
      print("\t{} already processed, skipping".format(ID))
      continue

    print("\tLooking for background data for tray {}...".format(tray))

    try:
      background = np.load(os.path.join(dump_path,'{}_background.npy'.format(tray)))
    except IOError:
      print("\t\tNo background file found for {}".format(tray))
      try:
        background = np.load(os.path.join(dump_path, '{}_background.npy'.format(ID)))
      except IOError:
        print("\t\tNo background file found for {}, skipping folder".format(ID))
        continue # Skip processing and proceed to next folder
    
    print("\tBackground data loaded")

    batched_files = split_data( glob(img_data_dir+'/*.npy'), 150 ) # batch size of 150

    batchIDs = ['{}-{}'.format(ID, i) for i in range(len(batched_files))]
    i = 0
    

    header = os.path.join(img_data_dir, 'header.txt')
    rmax = radius_from_res(max_res, header)

    for files in batched_files: 
      
      batchID = batchIDs[i]
      print('\tBatch nr: ', i)
      
      ############## read in data files ###############
      
      print("\t\tLoading image data...") 

      data = {f : np.load(f) for f in files}
      names = list(data.keys())
      
      ############   processing   ############
      
      print("\t\tPre-prossessing images...")
      
      process = Processor(data, background)

      process.rm_artifacts(value=500)
      process.background()
      data = process.processedData

      ##############  feature analysis  ##############
      
      print("\t\tExtracting profiles...")

      fex = Fex(data)

      fex.meanTraces(rmax=rmax, nangles=20)

      ############# saving ##############
      
      #print("Saving...")
      #process.dump_save(ID)

      print("\t\tSaving profiles to {}/{}_profiles.pickle...".format(dump_path, batchID))
      fex.dump_save(batchID, dump_path)
      
      del data
      del fex
      del process
      i += 1

    ############# join batches #############

    paths = [os.path.join(dump_path, '{}_profiles.pickle'.format(batchID)) for batchID in batchIDs]
    join_files(os.path.join(dump_path, '{}_profiles.pickle'.format(ID)), paths)

# obsolete
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
      rmax = radius_from_res(max_res, header)

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
      join_files('obsidian/datadump/{}new_profiles.pickle'.format(ID), paths)

if __name__ == '__main__':
  top = input("Enter top level directory: ")
  dump_path = 'obsidian/datadump/with-background'
  max_res = 7 #Angstrom
  pipe(top, dump_path, max_res)
