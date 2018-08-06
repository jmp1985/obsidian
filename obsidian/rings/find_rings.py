'''
User implementation of obsidian. For loading, analysing data and flagging images with protein rings
'''

from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from obsidian.learn.metrics import precision
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, make_frame
from obsidian.oimp.processor import Processor
from obsidian.fex.extractor import FeatureExtractor as Fex

# Assume cbf_tp_np stage completed, image_directories should contain a list of folders
# containing npy files
image_directories = ['/media/Elements/obsidian/diffraction_data/tray5/a3/grid']

#background_directory = ['/media/Elements/obsidian/diffraction_data/tray5/g1/grid']
background_directory = 'obsidian/datadump/tray5_background.npy'

background_data = np.load(background_directory)

#oimp = Processor()
#fex = Fex()

#data_table = df()

model = load_model('test-model.h5', custom_objects={'precision':precision})

data_dict = {'Data':['obsidian/datadump/T4f1_profiles.pickle'],
             'Class':['/media/Elements/obsidian/diffraction_data/classes/T4f1_classifications.pickle']}

for folder in image_directories:
  
  #dataframe = 'obsidian/datadump/database.pickle'
  #data = pickle_get(dataframe)[:2000]
  data = make_frame(data_dict)
  x = np.vstack(data['Data'].values)
  print("Searching for rings...")
  predictions = model.predict_proba(x.reshape(x.shape[0], x.shape[1], 1))
  data['Predicted'] = predictions

  if max(predictions)>0.5:
    print("Rings found!")
    print(np.array(predictions > 0.5).sum(),' / ',len(predictions))

  else:
    print("No rings found :(")
  top_five = data.sort_values('Predicted', ascending=False).iloc[:]
  print(top_five)

  #display_five = ImgDisp([np.load(f) for f in top_five['Path'].values])
  #display_five.disp()
  
  plt.figure()
  plotx = np.arange(len(top_five))
  truesx = np.where(top_five['Class'] == np.rint(top_five['Predicted']))[0]
  trues = top_five.iloc[top_five['Class'].values == np.rint(top_five['Predicted'].values)]
  falsesx = np.where(top_five['Class'] != np.rint(top_five['Predicted']))[0]
  falses = top_five.iloc[top_five['Class'].values != np.rint(top_five['Predicted'].values)]
  preds = top_five['Predicted'].values
  plt.plot(plotx, preds, truesx, trues['Class'].values, 'g.', falsesx, falses['Class'].values, 'r.')

  plt.show()
