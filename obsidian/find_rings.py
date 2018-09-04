'''
Process data and search for rings
'''

import os, sys, getopt, pickle, time
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from obsidian.learn.metrics import precision, weighted_binary_crossentropy
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import make_frame
from obsidian.oimp.oimp_main import pipe

def find_rings(model, data_frame, show_top=10, display_top=False, name=''):
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

  pd.set_option('max_colwidth', -1)
  with open('full_results.txt', 'a+') as f:
    f.write("\n\nClassification results for {}:\n".format(name))
    f.write(sort[['Path', 'Predicted']].to_string())

  with open('summary.txt', 'a+') as f:
    f.write("\n\nTop {} images for {}:\n".format(show_top, name))
    f.write(top[['Path', 'Predicted']].to_string())
 
  print("Top {} images:".format(show_top))
  print(top[['Path','Predicted']])

  if display_top:
    display = ImgDisp([np.load(f) for f in sort['Path'].values][:show_top])
    fig, ax = display.disp()
    for i in range(show_top):
      title = os.path.splitext(os.sep.join(sort.iloc[i]['Path'].split(os.sep)[-3:]))[0]
      ax.flat[i].set_title(title)
    plt.suptitle(os.path.splitext(name)[0])

def main(argv):
  '''
  Command line options:

  * -d: data directory containing images to be classified
  * -t: top n images to display / save to summary.txt
  * --display: if specified, display top n images to desktop for each sample
  '''

  start = time.time()
  
  data_directory = ''
  find_kwargs = {}
  model_name = 'standard_model'
  try:
    opts, args = getopt.getopt(argv, 'd:t:', ['display', 'model_name='])
  except getopt.GetoptError as e:
    print(e)
  for opt, arg in opts:
    if opt=='-d':
      data_directory = arg
    elif opt=='-t':
      find_kwargs['show_top'] = int(arg)
    elif opt=='--display':
      find_kwargs['display_top'] = True
    elif opt=='--model_name':
      model_name = arg

  dump = 'datadump'
  if not os.path.exists(dump):
    os.mkdir(dump)

  if not data_directory:
    data_directory = input("Enter data directory: ")
#'/media/Elements/obsidian/diffraction_data/lysozyme_small/tray2'

  max_res = 7 # Angstrom

  pipe(data_directory, dump, max_res)

# now _profiles.pickle files are in dump

  print("Loading model...")
  
  loss = weighted_binary_crossentropy(weight=0.5)
  
  model_path = os.path.join(os.path.dirname(__file__), learn, models, '{}.h5'.format(model_name))
  model = load_model(model_path, 
                      custom_objects={'precision':precision, 'weighted_loss':loss})

  #input_data = make_frame({'Data':glob(os.path.join(dump, '*profiles.pickle'))})
  
  for datablock in glob(os.path.join(dump, '*profiles.pickle')):
    print("Analysing {}".format(datablock))
    find_rings(model, make_frame({'Data':[datablock]}), name=datablock, **find_kwargs)

  end = time.time()
  secs = end - start
  print("Execution time: {0:d}:{1:.0f} min".format(int(secs//60), secs%60))

  plt.show()

if __name__=='__main__':
  main(sys.argv[1:])
