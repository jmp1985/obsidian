'''
main and experimental program for the machine learning stage of obsidian
'''

import os.path
import pickle
import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

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

# this is ugly and needs fixing
blocks = {1:('a2', 'a4', 'g1'), 2:('a4','a5','a7')}
IDs = ['T{}{}'.format(tray, well) for tray in blocks.keys() for well in blocks[tray]]

# locations of input data and labels
path1 = 'obsidian/datadump/{}_profiles.pickle'
path2 = '/media/Elements/obsidian/diffraction_data/classes/{}_classifications.pickle' 

# populate inputs and labels lists
profiles = []
classes = []
for ID in IDs:
  prof = pickle_get(path1.format(ID))
  cls = pickle_get(path2.format(ID))
  
  for name in cls.keys():
    profiles.append(prof[name])
    classes.append([cls[name]])

###########################
# massage data for keras  #
###########################
print(len(profiles))
print(len(classes))

# combine and shuffle inputs and targets
data = np.hstack((profiles, classes))
np.random.shuffle(data)

# reshape because keras demands it
data = data.reshape(data.shape[0], data.shape[1], 1)

# split data into seperate train and test sets
split = int(round(0.8*len(data)))
train_X, train_y = data[:split, :-1], data[:split, -1]
test_X, test_y = data[split:, :-1], data[split:, -1]

###################
#  display data   #
###################
''' these are just random sanity checks '''

print(data.shape)

demo = data[3,:-1] 
#plt.imshow(demo, cmap='gray')
print(demo,' ',data[3,-1])
plt.plot(demo)
plt.show()

print(demo.shape, type(demo))
demo = demo.reshape((demo.shape[0], 1))
print(demo, demo.shape)
print((data[:,-1]==1).sum())

###########################
#  build neural network   #
###########################

model = Sequential()

model.add(Conv1D(filters = 10, kernel_size = 5, activation='relu', input_shape=(2463, 1)))
model.add(MaxPooling1D())
model.add(Conv1D(filters = 50, kernel_size = 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(filters = 70, kernel_size = 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(filters = 70, kernel_size = 5, activation='relu'))
model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

plot_model(model, to_file='demo_model.png')

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#################
#   train net   #
#################

model.fit(train_X, train_y, epochs=10, batch_size=20)

score = model.evaluate(test_X, test_y, batch_size=20)

print(score)
