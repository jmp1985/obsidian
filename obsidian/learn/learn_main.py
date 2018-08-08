'''
Main and experimental program for the machine learning stage of obsidian
'''

import os.path
import pickle
import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential, load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obsidian.utils.imgdisp import ImgDisp
from obsidian.learn.convnet import ProteinClassifier
from sklearn.metrics import confusion_matrix

def pickle_get(path):
  '''
  Method encapsulating the pickle input process to tidy things up
  :param path: path with file to read into object
  '''
  pickle_in = open(path, 'rb')
  return pickle.load(pickle_in)


###################################
# import data and classifications #
###################################

def load_and_save():
# is ugly and needs fixing
  blocks = {1:( 'a2', 'a4','a6', 'a8', 'g1'), 2:('a4','a5','a7', 'a1', 'a2', 'a3','a8', 'g1'), 4:('a1-2','a2', 'a3', 'a4','a5', 'f1',), 5:('a1', 'a2', 'g1','f1')}
  IDs = ['T{}{}'.format(tray, well) for tray in blocks.keys() for well in blocks[tray]]

# locations of input data and labels
  path1 = 'obsidian/datadump/{}_profiles.pickle'
  path2 = '/media/Elements/obsidian/diffraction_data/classes/{}_classifications.pickle' 

  print("Gathering data...")
# populate inputs and labels lists
  profiles = []
  classes = []
  names = []
  for ID in IDs:
    prof = pickle_get(path1.format(ID))
    cls = pickle_get(path2.format(ID))
    
    for name in cls.keys():
      names.append(name)
      profiles.append(prof[name])
      classes.append(cls[name])

# DataFrame providing each image with a reference index
  lookup_table = pd.DataFrame({'Path' : names, 'Data' : profiles, 'Class' : classes})

  pickle.dump(lookup_table, open('obsidian/datadump/database.pickle', 'wb'))
  return profiles, classes, lookup_table

def load_frame(path='obsidian/datadump/database.pickle') :
  lookup_table = pickle_get(path)
  return list(lookup_table['Data']), list(lookup_table['Class']), lookup_table

def main():
  profiles, classes, lookup_table = load_and_save()

###########################
# massage data for keras  #
###########################

  assert len(profiles) == len(classes)

# combine and shuffle inputs and targets
  indexed_data = np.column_stack((list(lookup_table.index.values), profiles, classes))
  print(indexed_data.shape)
  np.random.shuffle(indexed_data)

# reshape because keras demands it
  data = indexed_data[:,1:].reshape(indexed_data.shape[0], indexed_data.shape[1]-1, 1)

# split data into seperate train and test sets
  split = int(round(0.8*len(data)))

  train_X, train_y = data[:split, :-1], data[:split, -1]
  test_X, test_y = data[split:, :-1], data[split:, -1]

# Retain indexes for tracking
  indexed_traindata = indexed_data[:split]
  indexed_testdata = indexed_data[split:]

###################
#  display data   #
###################
  ''' these are just random sanity checks '''

  print(data.shape)

  demo = data[3,:-1] 
#plt.imshow(demo, cmap='gray')
  print(demo,' ',data[3,-1])

  demo = demo.reshape((demo.shape[0], 1))
  print('Proportion of data with class 1: ',(data[:,-1]==1).sum()/len(data))

###########################
#  build neural network   #
###########################
  new_model = True

  if new_model:
    model = build_model()

    print(model.summary())

    #################
    #   train net   #
    #################

    history = model.fit(train_X[:], train_y[:], validation_data=(test_X, test_y), epochs=50, batch_size=20)
    model.save('test-model.h5')

    try:
      pickle.dump(history.history, open('obsidian/datadump/history.pickle','wb'))
    except Exception as e:
      print("History pickle failed: \n"+e)
    history = history.history

  else:
# Load prebuilt model
    model = load_model('test-model.h5')
    history = pickle.load(open('obsidian/datadump/history.pickle','rb'))

  score = model.evaluate(test_X, test_y, batch_size=20)
  fig0, ax0 = plot_performance(history)

  print('Score: ',score)

  guess = np.squeeze(model.predict_classes(test_X))
  probs = np.squeeze(model.predict_proba(test_X))
  np.set_printoptions(precision=3, suppress=True)


#############################
# analyse wrong predictions #
#############################

  wrong_predictions = indexed_testdata[guess != indexed_testdata[:,-1]]
  num = len(wrong_predictions)
  print(wrong_predictions, '\nNumber of wrong predictions: ',num, '/',len(guess))

  wrongs = [np.load(path)[1100:1550,1000:1500] for path in lookup_table.loc[wrong_predictions[:,0]]['Path']]
  show_wrongs(wrongs, wrong_predictions, probs)

  with pd.option_context('display.max_colwidth', 100):
    print(lookup_table.loc[wrong_predictions[:,0]][['Path', 'Class']])

  cm = confusion_matrix(test_y, guess)
  show_confusion(cm, [0, 1])

  plt.show()

if __name__ == "__main__":
  main()
