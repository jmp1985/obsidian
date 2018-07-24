'''
main and experimental program for the machine learning stage of obsidian
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
blocks = {1:( 'a2', 'a4','a6', 'a8', 'g1'), 2:('a4','a5','a7', 'a1', 'a2', 'a3','g1'), 4:('f1',), 5:('g1','f1')}
IDs = ['T{}{}'.format(tray, well) for tray in blocks.keys() for well in blocks[tray]]

# locations of input data and labels
path1 = 'obsidian/datadump/{}_profiles.pickle'
path2 = '/media/Elements/obsidian/diffraction_data/classes/{}_classifications.pickle' 

# populate inputs and labels lists
profiles = []
classes = []
names = []
for ID in IDs:
  print(ID)
  prof = pickle_get(path1.format(ID))
  cls = pickle_get(path2.format(ID))
  
  for name in cls.keys():
    names.append(name)
    profiles.append(prof[name])
    classes.append([cls[name]])

# DataFrame providing each image with a reference index
lookup_table = pd.DataFrame({'Path' : names, 'Data' : profiles, 'Class' : classes})

print(lookup_table)

###########################
# massage data for keras  #
###########################
assert len(profiles) == len(classes)
print(len(list(lookup_table.index.values)), len(profiles))
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
'''
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
'''
#################
#   train net   #
#################
'''
model.fit(train_X[:1000], train_y[:1000], epochs=10, batch_size=20)
model.save('test-model.h5')
'''
model = load_model('test-model.h5')
print(model.summary())

score = model.evaluate(test_X, test_y, batch_size=20)

print(score)

guess = np.squeeze(model.predict_classes(test_X))
probs = np.squeeze(model.predict_proba(test_X))
np.set_printoptions(precision=3, suppress=True)

#############################
# analyse wrong predictions #
#############################

wrong_predictions = indexed_testdata[guess != indexed_testdata[:,-1]]
num = len(wrong_predictions)
print(wrong_predictions, '\nNumber of wrong predictions: ',num, '/',len(guess))

wrongs = ImgDisp([np.load(path+'.npy')[1100:1550,1000:1500] for path in lookup_table.loc[wrong_predictions[:,0]]['Path']])

fig1, ax1 = wrongs.disp()
fig1.subplots_adjust(wspace=0.02, hspace=0.02)

for i in range(num):
  ax1.flat[i].set_title(wrong_predictions[i,-1])

with pd.option_context('display.max_colwidth', 100):
  print(lookup_table.loc[wrong_predictions[:,0]][['Path', 'Class']])

fig2, ax2 = plt.subplots(num)
for i in range(num):
  ax2[i].plot(lookup_table.loc[wrong_predictions[:,0]]['Data'].tolist()[i])
plt.show()

print()
