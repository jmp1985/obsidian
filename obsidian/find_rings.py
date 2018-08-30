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

  if display_top:
    pd.set_option('max_colwidth', -1)
    print("Top {} images:".format(show_top))
    print(sort[['Path','Predicted']].iloc[:show_top])
    display = ImgDisp([np.load(f) for f in sort['Path'].values][:show_top])
    fig, ax = display.disp()
    for i in range(show_top):
      title = os.path.splitext(os.sep.join(sort.iloc[i]['Path'].split(os.sep)[-3:]))[0]
      ax.flat[i].set_title(title)
    plt.suptitle(os.path.splitext(name)[0])

def main():

  start = time.time()

  dump = 'datadump'
  if not os.path.exists(dump):
    os.mkdir(dump)

  data_directory = '/media/Elements/obsidian/diffraction_data/lysozyme_small/tray2'

  max_res = 7 # Angstrom

  pipe(data_directory, dump, max_res)

# now _profiles.pickle files are in dump

  print("Loading model...")
  
  model_path = os.path.join(os.path.dirname(__file__), 'datadump', 'test-model.h5')
  model = load_model(model_path, 
                      custom_objects={'precision':precision, 'my_loss':weighted_binary_crossentropy})

  #input_data = make_frame({'Data':glob(os.path.join(dump, '*profiles.pickle'))})
  
  for datablock in glob(os.path.join(dump, '*profiles.pickle')):
    print("Analysing {}".format(datablock))
    find_rings(model, make_frame({'Data':[datablock]}), show_top=10, display_top=True, name=datablock)

  end = time.time()
  secs = end - start
  print("Execution time: {0:d}:{1:.0f} min".format(int(secs//60), secs%60))

  plt.show()

if __name__=='__main__':
  main()
