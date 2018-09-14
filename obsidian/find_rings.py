'''
Process data and search for rings
'''

import os, sys, getopt, pickle, time, datetime
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import make_frame
from obsidian.oimp.oimp_main import pipe, get_img_dirs
from obsidian.learn.metrics import precision, weighted_binary_crossentropy

def find_rings(model, data_frame, show_top=10, display_top=False, name=''):
  '''Feed collection of data to a pretrained model, predict probabilities of the presence 
  of protein rings and display most likely candidates

  :param pd.DataFrame data_frame: database to be analysed, containing 'Path' and 'Data' columns
  :param model: keras pretrained model
  :param int show_top: n most highly rated images to display
  :param bool display_top: if True, display highest ranging images graphically
  :param str name: datablock title
  '''
  # Extract input data from table
  x = np.vstack(data_frame['Data'].values)
  
  print("Searching for rings...")

  # Predict probabilities for datablock
  predictions = model.predict_proba(np.expand_dims(x, 2))
  data_frame['Predicted'] = predictions
  
  if max(predictions) > 0.5:
    print("Rings found!")
    print('in ',np.array(predictions > 0.5).sum(),' images out of ',len(predictions))
  
  else:
    print("No rings found :(")

  # Rank images by prediction
  sort = data_frame.sort_values('Predicted', ascending=False)
  top = sort.iloc[:show_top]
  
  # Write results to txt files
  pd.set_option('max_colwidth', -1)
  with open('full_results.txt', 'a+') as f:
    f.write("\n\nClassification results for {}:\n".format(name))
    f.write(sort[['Path', 'Predicted']].to_string())

  with open('summary.txt', 'a+') as f:
    f.write("\n\nResults for {}: \n".format(name))
    f.write("Rings found in "+
            str(np.array(predictions > 0.5).sum())+
            ' images out of '+
            str(len(predictions)))
    f.write("\n\nTop {} images:\n".format(show_top))
    f.write(top[['Path', 'Predicted']].to_string())

  # Also print results to screen
  print("Top {} images:".format(show_top))
  print(top[['Path','Predicted']])

  # Display highest ranking images
  if display_top:
    display = ImgDisp([np.load(f) for f in sort['Path'].values][:show_top])
    fig, ax = display.disp()
    for i in range(show_top):
      title = os.path.splitext(os.sep.join(sort.iloc[i]['Path'].split(os.sep)[-3:]))[0]
      ax.flat[i].set_title(title)
    plt.suptitle(os.path.splitext(name)[0])

def one_by_one(model, data_directory, dump, max_res, find_kwargs):
  '''Process each grid scan and classify in turn, before moving
  onto next image folder. After each folder, the created data file will
  be deleted.

  :param model: trained keras classifier
  :param str data_directory: top level image directory containing grid scans
  :param str dump: local storage directory for intermediate files
  :param float max_res: maximum resultion to which images are cut / to extract data from
  '''
  
  for folder in get_img_dirs(data_directory):
    #print("\nProcessing {}...".format(folder))
    datablock = pipe(folder, dump, max_res)
    if datablock: # Unsuccessfull processing will return empty opject
      find_rings(model, make_frame({'Data':[datablock]}), name=datablock, **find_kwargs)
      os.remove(datablock)

def all_at_once(model, data_directory, dump, max_res, find_kwargs):
  '''Process all data first and then classify each datablock (grid_scan) in turn.

  :param model: trained keras classifier
  :param str data_directory: top level image directory containing grid scans
  :param str dump: local storage directory for intermediate files
  :param float max_res: maximum resultion to which images are cut / to extract data from
  '''
  print("\nProcessing data...\n")
  
  # Preprocess images and feature extract, saving xx_profiles.pickle files in dump
  pipe(data_directory, dump, max_res)
  
  # Classify data
  for datablock in glob(os.path.join(dump, '*profiles.pickle')):
    print("\n###### Analysing {} ######".format(datablock))
    find_rings(model, make_frame({'Data':[datablock]}), name=datablock, **find_kwargs)
 
def main(argv):
  '''Command line options:

  * -d: data directory containing images to be classified
  * -t: top n images to display / save to summary.txt
  * --display: if specified, display top n images to desktop for each sample
  '''

  ############## Setting up ############

  start = time.time()
  
  data_directory = ''
  find_kwargs = {}
  model_name = 'standard_model'
  delete = False
  max_res = None

  help_message = 'find_rings\
    -h (display this message)\
    -d <data directory> (required)\
    -t <n top images to show>\
    -r <maximum resolution>\
    --display (display top n images graphically)\
    --model_name <name of specific model to use for classification>\
    --delete (process data folders one by one and delete intermediate files)'
  try:
    opts, args = getopt.getopt(argv, 'd:t:hr:', ['display', 'model_name=', 'delete'])
  except getopt.GetoptError as e:
    print(e)
    print(help_message)
    sys.exit(2)
  for opt, arg in opts:
    if opt=='-h':
      print(help_message)
      sys.exit()
    elif opt=='-d':
      data_directory = os.path.abspath(arg)
    elif opt=='-t':
      find_kwargs['show_top'] = int(arg)
    elif opt=='--display':
      find_kwargs['display_top'] = True
    elif opt=='--model_name':
      model_name = arg
    elif opt=='--delete':
      delete = True
    elif opt=='-r':
      max_res = arg
      
  dump = 'datadump'
  if not os.path.exists(dump):
    os.mkdir(dump)

  # Enter data directory if not provided as argument
  if not data_directory:
   print("!! Argument -d required !!\n", help_message)
   sys.exit(2)

  with open('summary.txt', 'w') as f:
    f.write("\n{}\nRunning obsidian on data in {}".format('#'*60, data_directory))
    f.write("\nUsing model: {}".format(model_name))
    f.write("\nTime: {}".format(str(datetime.datetime.now())))

  ############## Load Model ################

  print("\nLoading model...")
  
  from keras.models import Sequential, load_model

  models_dir = os.path.join(os.path.dirname(__file__), 'learn', 'models')

  # Load custom loss parameters used in model training
  with open(os.path.join(models_dir, '{}.txt'.format(model_name))) as f:
    params = {line.split(' ')[0] : line.split(' ')[1] for line in f}

  loss_weight = eval(params['loss_weight'])
  try:
    loss = eval(params['loss'])
  except NameError:
    loss = params['loss']

  model_path = os.path.join(models_dir, '{}.h5'.format(model_name))
  model = load_model(model_path, 
                      custom_objects={'precision':precision, 'weighted_loss':loss})
  
  ############# Process Data #############

  if delete:
    one_by_one(model, data_directory, dump, max_res, find_kwargs)
  else:
    all_at_once(model, data_directory, dump, max_res, find_kwargs)

  # Time process
  end = time.time()
  secs = end - start
  print("Execution time: {0:d}:{1:.0f} min".format(int(secs//60), secs%60))

  plt.show()

if __name__=='__main__':
  main(sys.argv[1:])
