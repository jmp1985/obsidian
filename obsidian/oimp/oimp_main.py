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

def get_img_dirs(root):
  '''Walk through root directory compiling a list of all bottom level directories
  (that is directories that contain only files and no subdirectories)

  :param str root: directory to walk through
  :returns: dictionary of image data directories with generated IDs based local directory tree
  '''
  bottom_dirs = {}
  for dirname, subdirList, fileList in os.walk(root, topdown=False):
    if len(subdirList)==0:
      ID = ''.join(dirname.split(os.sep)[-3:])
      bottom_dirs[dirname] = ID
      
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
  '''Takes a single top level directory as input and processes all nested files.
  IDs are derived from directory paths. Background files are searched for in a bottom-up manner.

  :param str top: Top level directory containing all image files (which may be in several subdirectories)
  :param str dump_path: Directory to dump processed data into
  '''
 
  last = ''
  
  # Find all relevant image directories within top
  bottom_dirs = get_img_dirs(top)

  # Process each image directory in turn
  for img_data_dir in bottom_dirs.keys():
    
    assert os.path.exists(img_data_dir), "{} not found".format(img_data_dir)
    ID = bottom_dirs[img_data_dir]
    print("\n###### Working on {}... ######".format(ID))  
    
    # Skip directory if already processed
    if os.path.exists(os.path.join(dump_path, '{}_profiles.pickle'.format(ID))):
      print("\t{} already processed, skipping".format(ID))
      continue

    # Background file
    print("\tLooking for background data for {}...".format(img_data_dir))

    background_path = locate_background_file(img_data_dir)
    
    if background_path is None:
      print("\t\tNo background file found for {}, skipping folder".format(ID))
      continue
    else:
      background = np.load(background_path)

    print("\tBackground data loaded from: {}".format(background_path))

    # Batch data processing to avoid memory issues
    batched_files = split_data( glob(img_data_dir+'/*.npy'), 400 ) # batch size of 400

    batchIDs = ['{}-{}'.format(ID, i) for i in range(len(batched_files))]
    i = 0

    # Extract max radius in pixels from image header
    header = os.path.join(img_data_dir, 'header.txt')
    rmax = radius_from_res(max_res, header) if max_res is not None else None

    # Open keys file to link each image with unique original image path
    with open(os.path.join(img_data_dir, 'keys.txt')) as k:
      keys = {line.split()[0] : line.split()[1] for line in k}

    for files in batched_files: 
      
      batchID = batchIDs[i]
      print('\tBatch nr: ', i)
      
      ############## Read in data files ###############
      
      print("\t\tLoading image data...") 

      data = {f : np.load(f) for f in files if 'background' not in f}
      
      ############   Processing   ############
      
      print("\t\tPre-prossessing images...")
      
      process = Processor(data, background)

      process.rm_artifacts(value=500)
      process.background()
      data = process.processedData

      ##############  Feature analysis  ##############
      
      print("\t\tExtracting profiles...")

      fex = Fex(data)
      mean_traces = fex.mean_traces(rmax=rmax, nangles=20)

      ############# Saving ##############

      # Create an indexed dictionary with the keys derived from keys.txt
      indexed_data = {keys[path] : {'Path' : path, 'Data' : mean_traces[path]} for path in mean_traces }
      
      print("\t\tSaving profiles to {}/{}_profiles.pickle...".format(dump_path, batchID))
      pickle_put(os.path.join(dump_path, '{}_profiles.pickle'.format(batchID)), indexed_data), 
      #fex.dump_save(batchID, dump_path)
      
      del data
      del fex
      del process
      i += 1

    ############# Join batches to single file #############

    paths = [os.path.join(dump_path, '{}_profiles.pickle'.format(batchID)) for batchID in batchIDs]
    last = os.path.join(dump_path, '{}_profiles.pickle'.format(ID))
    join_files(last, paths)
  
  # Return path of last processed file
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
