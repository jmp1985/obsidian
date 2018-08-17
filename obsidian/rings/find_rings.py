'''
User implementation of obsidian. For loading, analysing data and flagging images with protein rings
'''
import os, sys, getopt, pickle, time
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from obsidian.learn.metrics import precision, weighted_binary_crossentropy
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, make_frame, split_data, join_files, read_header
from obsidian.oimp.processor import Processor
from obsidian.fex.extractor import FeatureExtractor as Fex, radius_from_res

def find_rings(model, data_frame, show_top=10, display_top=False, classified=False):
  '''
  Feed collection of data to a pretrained model, predict probabilities of the presence of protein 
  rings and display most likely candidates

  :param pd.DataFrame data_frame: database to be analysed, containing 'Path' and 'Data' columns
  :param model: keras pretrained model
  :param int show_top: n most highly rated images to display
  '''
  # Extract input data from table
  x = np.vstack(data_frame['Data'].values)
  
  print("Searching for rings...")
  predictions = model.predict_proba(np.expand_dims(x, 2))
  data_frame['Predicted'] = predictions
  
  if max(predictions) > 0.5:
    print("Rings found!")
    print('in ',np.array(predictions > 0.5).sum(),' images out of ',len(predictions))
  
  else:
    print("No rings found :(")
   
  sort = data_frame.sort_values('Predicted', ascending=False)
  top = sort.iloc[:show_top]

  if display_top:
    pd.set_option('max_colwidth', -1)
    print("Top {} images:".format(show_top))
    print(sort[['Path','Predicted']].iloc[:show_top])
    display = ImgDisp([np.load(f) for f in sort['Path'].values].iloc[:show_top])
    print("Borderline images:")
    print(sort.iloc[np.rint(sort['Predicted'].values) == 1].iloc[-5:])
    display.disp()

    if classified:
      plt.figure()
      plotx = np.arange(len(sort))
      truesx = np.where(sort['Class'] == np.rint(sort['Predicted']))[0]
      trues = sort.iloc[sort['Class'].values == np.rint(sort['Predicted'].values)]
      falsesx = np.where(sort['Class'] != np.rint(sort['Predicted']))[0]
      falses = sort.iloc[sort['Class'].values != np.rint(sort['Predicted'].values)]
      preds = sort['Predicted'].values
      plt.plot(plotx, preds, truesx, trues['Class'].values, 'g.', falsesx, falses['Class'].values, 'r.')
      print("Wrong images: \n", falses[['Path', 'Predicted']])
      wrongs = ImgDisp([np.load(f) for f in falses['Path'].values])
      wrongs.disp()
      
def process(image_directories, background_files, IDs, direct, nangles=20, process=False, background=False, max_res=7):
  '''
  Prepare data for classification buy running it through obsidian image processing and feature extraction functions

  :param list image_directories: list of directories containing raw image files in npy format
  :param list background_files: list of npy background files corresponding to the image directories
  :param list IDs: list of identification strings for each image set
  :param str direct: destination folder for intermediate processing files
  :param int nangles: number of line profiles to extract from each image
  :param bool process: if True, preprocess images before feature extraction
  :param bool background: if True, perform background subtraction as part of image processing
  '''
  def do_the_processing():
    '''
    '''
    if background:
      bg = np.load(bg_file)
    
    batched_files = split_data( glob(folder+'/*npy'), 150  )
    n = 0
    # Extract header parameters
    header = os.path.join(folder,'header.txt')
    
    # Determine max_radius in pixels for line data extraction
    try:
      rmax = radius_from_res(max_res, header)
    except Exception as e:
      print("Failed to extract necessary image parameters. Enter 'y' to continue program with following defaults: \nLambda: 0.96863 A \nDistance: 0.49906 m \nPixel size: 172e-6 m")
      if input("Continue? [y/n]") != 'y':
        print(e)
        sys.exit('Program terminated')
      else:
        pass # exception handled 
    
    for files in batched_files:
      
      batchID = n
      print("Loading batch...")
      data = {f : np.load(f) for f in files}
      if process:
        p = Processor(data)
        p.rm_artifacts(value = 500)
        if background:
          p.background(bg)
        data = p.processedData
        del p
      print("Extracting profiles from batch {}".format(n))
      fex = Fex(data)
      fex.meanTraces( rmax=rmax, nangles=nangles)
      fex.dump_save(batchID, path=direct)
      
      del data
      del fex
      n += 1
    
    paths = [ os.path.join(direct, '{}_profiles.pickle'.format(batch_nr)) for batch_nr in range(n) ]
    join_files( os.path.join(direct, '{}_profiles.pickle'.format(ID)), paths )
  
  if background:
    assert background_files is not None, "Provide background files first"
    for folder, bg_file, ID in zip(image_directories, background_files, IDs):
      print("Processing {}".format(ID))
      do_the_processing()
  else:
    for folder, ID in zip(image_directories, IDs):
      print("Processing {}".format(ID))
      do_the_processing()

def main(argv):
  start = time.time()
  
  # Parse command line options
  
  kwargs = {}
  classified = False
  try:
    opts, args = getopt.getopt(argv, 'a:cbp')
  except GetoptError as e:
    print(e)
    sys.exit(2)

  for opt, arg in opts:
    if opt=='-a':
      kwargs['nangles'] = int(arg)
    if opt=='-b':
      kwargs['background'] = True
    if opt=='-p':
      kwargs['process'] = True
    if opt=='-c':
      classified = True

  direct = 'pipe_test'
  if not os.path.exists(direct):
    os.mkdir(direct)

  # Assume cbf_tp_np stage completed, image_directories should contain a list of folders
  # containing npy files
  image_directories = ['/media/Elements/obsidian/diffraction_data/180726_small/tray1/a5']

  #background_directories = ['/media/Elements/obsidian/diffraction_data/tray5/g1/grid']
  background_files = ['/dls/science/users/ywl28659/obsidian/obsidian/datadump/180726_smalltray1a5_background.npy']

  # IDs
  IDs = ['180726_smalltray1a5']

  print("Loading model")
  model = load_model('obsidian/datadump/test-model.h5', custom_objects={'precision':precision, 'my_loss':weighted_binary_crossentropy})

  # Data processing and feature extraction
  process(image_directories, background_files, IDs, direct, **kwargs)
  # Processed data now saved in <direct> as <ID>_profiles.pickle
  
  if classified:
    data_dict = {'Data':[os.path.join(direct, '{}_profiles.pickle'.format(ID)) for ID in IDs],
                 'Class':['/media/Elements/obsidian/diffraction_data/classes/{}_classifications.pickle'.format(ID) for ID in IDs]}
  else:  
    data_dict = {'Data':[os.path.join(direct, '{}_profiles.pickle'.format(ID)) for ID in IDs]}

  data = make_frame(data_dict, classified)

  find_rings(model, data, classified=classified, display_top=True)

  end = time.time()
  secs = end - start
  print("Execution time: {0:d}:{1:.0f} min".format(int(secs//60), secs%60))

  plt.show()

if __name__=='__main__':
  main(sys.argv[1:])
