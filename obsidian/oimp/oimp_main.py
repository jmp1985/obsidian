'''
Image processing main class for testing and developing
This module brings together the image processing and line extracion functions of obsidian. 
The directories containing files to be processed are taken
from user input (exactly how differs between main1 and main2) and the 
files are then passed through Processor and FeatureExtractor objects respectively, and the resulting
data dictionaries of form {imagepath:mean_profile_data} are saved to pickle files named with IDs
'''
import os, os.path, sys, getopt
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, pickle_put, join_files, split_data, read_header
from obsidian.utils.build_bg_files import bg_from_scan
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

def locate_background_file(img_data_dir):
  '''Ascend img_dir until the first candidate for a background
  file is found.
  
  :param str img_data_dir: bottom directory of diffraction images
  :returns: path to background file
  '''
  current_path = img_data_dir

  while True:
    next_path = os.path.realpath(os.path.join(current_path, '..'))
    if 'background.npy' in os.listdir(current_path):
      return os.path.join(current_path, 'background.npy')
    elif next_path == current_path:
      return None
    else:
      current_path = next_path

def pipe(top, dump_path, max_res):
  '''
  This version takes a single top level directory as input and processes all nested files.
  IDs are derived from directory paths. Background files are created if not already existing
  '''
  ##################
  #   directories  #
  ##################
  
  last = ''

  bottom_dirs = get_img_dirs(top)
  #bg_from_scan(top, dump_path, ('f1', 'g1'))
  for img_data_dir in bottom_dirs.keys():
  ############################
  # read in background data  #
  ############################
     
    assert os.path.exists(img_data_dir), "{} not found".format(img_data_dir)
    ID = bottom_dirs[img_data_dir]['ID']
    tray = bottom_dirs[img_data_dir]['tray']
    print("\n###### Working on {}... ######".format(ID))  
    
    if os.path.exists(os.path.join(dump_path, '{}_profiles.pickle'.format(ID))):
      print("\t{} already processed, skipping".format(ID))
      continue

    print("\tLooking for background data for {}...".format(img_data_dir))

    background_path = locate_background_file(img_data_dir)
    
    if background_path is None:
      print("\t\tNo background file found for {}, skipping folder".format(ID))
      continue
    else:
      background = np.load(background_path)

    print("\tBackground data loaded from: {}".format(background_path))

    # Batch data processing to avoid memory issues
    batched_files = split_data( glob(img_data_dir+'/*.npy'), 400 ) # batch size of 150

    batchIDs = ['{}-{}'.format(ID, i) for i in range(len(batched_files))]
    i = 0

    header = os.path.join(img_data_dir, 'header.txt')
    rmax = radius_from_res(max_res, header) if max_res is not None else None

    for files in batched_files: 
      
      batchID = batchIDs[i]
      print('\tBatch nr: ', i)
      
      ############## read in data files ###############
      
      print("\t\tLoading image data...") 

      data = {f : np.load(f) for f in files if 'background' not in f}
      
      ############   processing   ############
      
      print("\t\tPre-prossessing images...")
      
      process = Processor(data, background)

      process.rm_artifacts(value=500)
      process.background()
      data = process.processedData

      ##############  feature analysis  ##############
      
      print("\t\tExtracting profiles...")

      fex = Fex(data)
      fex.mean_traces(rmax=rmax, nangles=20)

      ############# saving ##############
      
      print("\t\tSaving profiles to {}/{}_profiles.pickle...".format(dump_path, batchID))
      fex.dump_save(batchID, dump_path)
      
      del data
      del fex
      del process
      i += 1

    ############# join batches #############

    paths = [os.path.join(dump_path, '{}_profiles.pickle'.format(batchID)) for batchID in batchIDs]
    join_files(os.path.join(dump_path, '{}_profiles.pickle'.format(ID)), paths)
    last = os.path.join(dump_path, '{}_profiles.pickle'.format(ID))
  
  return last

def run(argv):
  
  top, dump, max_res = None, None, None
  # Parse command line options
  try:
    opts, args = getopt.getopt(argv, 't:d:r:')
  except getopt.GetoptError as e:
    print(e)                   
    sys.exit(2)
  for opt, arg in opts:
    if opt=='-t':
      top = arg
    elif opt=='-d':
      dump = arg
    elif opt=='-r':
      max_res = int(arg)

  if not all((top, dump)):
    print("-t, -d are required")
    sys.exit(2)

  if not os.path.exists(dump):
    os.makedirs(dump)

  pipe(top, dump, max_res)

if __name__ == '__main__':
  run(sys.argv[1:])
