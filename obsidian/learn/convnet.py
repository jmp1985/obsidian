'''
Module of methods for building, training and handling models
:
'''

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import itertools
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, pickle_put
from obsidian.learn.metrics import precision
class ProteinClassifier():
  
  '''
  Member variables:
  data_table
  profiles
  classes
  model
  indexed_data (shuffled)
  history
  split
  '''

  def __init__(self):
    self.data_table = None
    self.model = None
    self.indexed_data = None

  def load_table(self, path):
    self.data_table = pickle_get(path)
    self.profiles = list(self.data_table['Data'])
    self.classes = list(self.data_table['Class'])
    assert len(self.profiles) == len(self.classes)

  def massage(self):
    '''
    massage data into shape, prepare for model fitting
    '''
    assert self.data_table is not None, "Load data fist!" 

    # Combine and shuffle inputs and targets
    self.indexed_data = np.column_stack( (list(self.data_table.index.values), self.profiles, self.classes) )
    np.random.shuffle(self.indexed_data)
    
    # Once shuffled strip index and reshape for keras
    data = self.indexed_data[:,1:].reshape(self.indexed_data.shape[0], self.indexed_data.shape[1]-1, 1)
    self.shuffled_data = data

    # Split into train and test sets
    split_val = 0.8
    self.split = int(round(split_val*len(data)))
    
    # Last column of data contains class labels
    self.X_train, self.y_train = data[:self.split, :-1], data[:self.split, -1]
    self.X_test, self.y_test = data[self.split:, :-1], data[self.split:, -1]

  def print_summary(self):
    print("Data loaded!")
    print("Total number of samples: {0}\nBalance: class 1 - {1:.2f}%".format(len(self.profiles), (np.array(self.classes) == 1).sum()/len(self.classes)))
    print("Network to train:\n", self.model.summary())
    
  def build_model(self, nlayers=3, min_kern_size=5, max_kern_size=200, padding='same', dropout=0.3):
    
    kernel_sizes = np.linspace(min_kern_size, max_kern_size, nlayers, dtype=int)
    nfilters = np.linspace(20, 50, nlayers, dtype=int)
    
    model = Sequential()

    # Input layer
    model.add(Conv1D(filters = nfilters[0], kernel_size=kernel_sizes[0].item(), padding=padding, activation='relu', input_shape=(2463, 1)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))

    for i in range(1,nlayers):
      model.add(Conv1D(filters=nfilters[i], kernel_size=kernel_sizes[i].item(), padding=padding, activation='relu'))
      model.add(MaxPooling1D())
      model.add(Dropout(dropout))
    '''
    model.add(Conv1D(filters = 30, kernel_size = 10, padding = 'same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.4))
    
    model.add(Conv1D(filters = 40, kernel_size = 50, padding = 'same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.4))
  
    model.add(Conv1D(filters = 30, kernel_size = 200, padding = 'same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.4))
    '''
    model.add(Flatten())
    
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #optimiser
    #adam = Adam(lr=0.01, decay=0.5)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    self.model = model
    return model

  def train_model(self, end=-1, epochs=30, batch_size=20):
    '''
    Shuffle and split data, then feed to model
    '''
    assert self.model is not None, "Build model first!"
    
    self.massage()
    self.print_summary()

    self.history = self.model.fit(self.X_train[:end], self.y_train[:end], validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size).history
    
    self.model.save('test-model.h5')
    pickle_put('obsidian/datadump/history.pickle', self.history)
    
    # Plot training history
    self.plot_performance(self.history)

  def model_from_save(self):
    print("WARNING: expect inaccurate performance readings when testing pre-trained models on seen data")

    self.model = load_model('test-model.h5')
    self.history = pickle_get('obsidian/datadump/history.pickle')

    # Plot training history
    self.plot_performance(self.history) 

  def test_model(self, batch_size=20):
    score = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
    print("Loss: {0:.2f}, Accuracy: {1:.2f}".format(score[0], score[1]))
    
    # Analyse performance
    predicted = self.model.predict_classes(self.X_test)
    probs = self.model.predict_proba(self.X_test)

    cm = confusion_matrix(self.y_test, predicted)
    self.show_confusion(cm, [0,1])
    
    '''
    indexed_test = indexed_data[self.split:]
    wrong_predictions = indexed_test[ predicted != indexed_test[:,-1]]
    '''

  def grid_search(self):
    '''
    Perform a grid search of parameter space to find optimal hyperparameter configuration
    '''
    model = KerasClassifier(build_fn=self.build_model, batch_size=20, epochs=20, verbose=0)
    epochs = [35, 50]
    dropout = [0.2, 0.3, 0.5]
    max_kern_size = [50, 100, 200, 400]
    min_kern_size = [3, 5, 10]
    nlayers = [2, 3, 4]
    batch_size = [10, 20]
    padding = ['valid', 'same']
    
    param_grid = dict(epochs=epochs, dropout=dropout, max_kern_size=max_kern_size, nlayers=nlayers, batch_size=batch_size, padding=padding)
    #param_grid = dict(padding=padding)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=['accuracy', 'precision'], refit=False)
    search_result = grid.fit(self.shuffled_data[:,:-1], self.shuffled_data[:,-1])
    pickle_put('obsidian/datadump/grid_search.pickle', search_result)

  def plot_performance(self, history):
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
  
  def show_confusion(self, cm, classes):
    '''
    Display confusion plot
    :param cm: calcualted confusion matrix
    :classes: list of class labels
    '''
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color = 'black' if cm[i,j]<thresh else 'white')

    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    
  def show_wrongs(self, wrongs, wrong_predictions, probs):
    '''
    Display wongly classified images together with their true class and predicted class probability
    :param wrongs: paths of images
    :param wrong_predications: predicted classes
    :param probs: predicted probabilities
    '''
    num = len(wrongs)
    wrongs = ImgDisp(wrongs)

    fig1, ax1 = wrongs.disp()
    fig1.subplots_adjust(wspace=0.02, hspace=0.02)

    try:
      for i in range(num):
        ax1.flat[i].set_title((str(wrong_predictions[i,-1])+' '+str(probs[i])))
    except Exception as e:
      print("Couldn't label plots: \n", e)

def main():
  PC = ProteinClassifier()
  PC.load_table('obsidian/datadump/database.pickle')
  PC.massage()
  PC.grid_search()
  #PC.build_model(nlayers=3, min_kern_size=3, max_kern_size=40)
  #PC.train_model(end=100, epochs=10)
  #PC.test_model()
  plt.show()
  

if __name__ == '__main__':
  main()
