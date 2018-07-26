'''
Module of methods for building, training and handling models
'''

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt

def build_net():
  model = Sequential()

  model.add(Conv1D(filters = 10, kernel_size = 5, activation='relu', input_shape=(2463, 1)))
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Conv1D(filters = 50, kernel_size = 5, activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Conv1D(filters = 70, kernel_size = 10, activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))
  
  model.add(Conv1D(filters = 70, kernel_size = 15, activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(200, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
  return model

def plot_performance(history):
  '''
  :param history: keras history object containing loss and accuracty info from training
  '''
  
  fig, (ax1, x2) = plt.subplots(ncols=2)

  ax1.plot(history.history['loss'], label = 'train', color = '#ff335f')
  ax1.plot(history.history['val_loss'], label = 'validation', color = '#5cd6d6')
  ax1.xlabel('epoch')
  ax1.ylabel('loss')
  
  ax2.plot(history.history['acc'], label = 'train', color = '#ff335f')
  ax2.plot(history.history['val_acc'], label = 'validation', color = '#5cd6d6')
  ax2.xlabel('epoch')
  ax2.ylabel('accuracy')
  
  plt.legend(loc='upper centre' ,bbox_to_anchor=(0.5, -0.1))

  return fig, (ax1, ax2)
