'''
main and experimental program for the machine learning stage of obsidian
'''

import os.path
import pickle
import keras
from keras.layers import Dense, Conv1D, 
                         MaxPooling1D, Flatten, Dropout
from keras.models import Sequential
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

###################################
# import data and classifications #
###################################

classes_dir = 'data/realdata/npy_files/tray2/a5/grid'

# 1d mean traces or processed 2d images as input data
pickle_in = open('obsidian/datadump/profiles.pickle', 'rb')
#pickle_in = open('obsidian/datadump/processed.pickle', 'rb')
inputs = pickle.load(pickle_in)

# classifications
pickle_in = open(os.path.join(classes_dir, 'classifications.pickle'), 'rb')
classes = pickle.load(pickle_in)

#data = {name : {'x' : inputs[name], 'y' : classes[name]} for name in inputs.keys() }
data = np.array([[inputs[name], classes[name]] for name in inputs.keys()])
print(data.shape)
'''
input_data = np.laod('obsidian/datadump/a4data.npy')
print(input_data, input_data.shape)
'''
###########################
# massage data for keras  #
###########################



###################
# display data    #
###################

demo = data[3,0] 
#plt.imshow(demo, cmap='gray')
plt.plot(demo)
#plt.show()

demo = demo.reshape((demo.shape[0], 1))
print(demo.shape, type(demo))

###########################
#  build neural network   #
###########################

model = Sequential()

model.add(Conv1D(filters = 5, kernel_size = 5, input_shape=(2463, 1)))
print(model.output_shape)

model.add(MaxPooling1D())
print(model.output_shape)

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dense(2, activation='softmax'))

plot_model(model, to_file='demo_model.png')
