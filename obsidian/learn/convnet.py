'''
Module of methods for building, training and handling models
'''

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from obsidian.utils.imgdisp import ImgDisp

def build_net():
  model = Sequential()

  model.add(Conv1D(filters = 20, kernel_size = 5, padding = 'same', activation='relu', input_shape=(2463, 1)))
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Conv1D(filters = 50, kernel_size = 10, padding = 'same', activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.4))

  model.add(Conv1D(filters = 50, kernel_size = 50, padding = 'same', activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.4))
  
  model.add(Conv1D(filters = 70, kernel_size = 200, padding = 'same', activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.4))
  
  model.add(Flatten())
  model.add(Dense(250, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  #optimiser
  #adam = Adam(lr=0.01, decay=0.5)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model

def plot_performance(history):
  '''
  :param history: keras history object containing loss and accuracty info from training
  '''
  
  fig, (ax1, ax2) = plt.subplots(ncols=2)

  ax1.plot(history['loss'], label = 'train', color = '#ff335f')
  ax1.plot(history['val_loss'], label = 'validation', color = '#5cd6d6')
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('loss')
  
  ax2.plot(history['acc'], label = 'train', color = '#ff335f')
  ax2.plot(history['val_acc'], label = 'validation', color = '#5cd6d6')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('accuracy')
  
  plt.legend(loc='lower right')

  return fig, (ax1, ax2)

def show_confusion(cm, classes):
  
  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  
  thresh = cm.max()/2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color = 'black' if cm[i,j]<thresh else 'white')

  plt.xlabel('Predicted class')
  plt.ylabel('True class')

def show_wrongs(wrongs):
  
  wrongs = ImgDisp(wrongs)

  fig1, ax1 = wrongs.disp()
  fig1.subplots_adjust(wspace=0.02, hspace=0.02)

  try:
    for i in range(num):
      ax1.flat[i].set_title((str(wrong_predictions[i,-1])+' '+str(probs[i])))
  except Exception as e:
    print("Couldn't label plots: \n", e)


