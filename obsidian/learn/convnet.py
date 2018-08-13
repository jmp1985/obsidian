'''
Module of Classes and methods for building, training and handling classification models
'''

import sys, getopt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import itertools
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, pickle_put
from obsidian.learn.metrics import precision, weighted_binary_crossentropy
import pandas as pd
import pickle

class ProteinClassifier():
  '''
  Class for specifying, building, training and testing diffraction image classifiers

  :ivar data_table: database of all available images with image path, extracted data and class
  :ivar profiles: list of profiles
  :ivar classes: list of class labels
  :ivar model: Keras model object
  :ivar indexed_data: (shuffled)
  :ivar history: created when model trained
  :ivar split: fraction of data samples to be used for training
  
  **Default parameters:**
    | number of layers: 3
    | kernel sizes:     min 3, max 100
    | dropout:          0.3
    | padding:          same
    | number of epochs: 30
    | batch size:       20
  '''

  def __init__(self):
    '''
    When a ProteinClassifier object is instantiated, data_table, model and indexed_data are set to None to ensure that methods are called in the appropriate order
    '''
    self.data_table = None
    self.model = None
    self.indexed_data = None
  
  @staticmethod
  def make_database():
    '''
    '''
    # is ugly and needs fixing
    blocks = {1:( 'a2', 'a4','a6', 'a8', 'g1'), 2:('a4','a5','a7', 'a1', 'a2', 'a3','a8', 'g1'), 4:('a1-2','a2', 'a3', 'a4','a5', 'f1',), 5:('a1','a2','g1','f1'), 6:('a1-2', 'a2-1', 'a3')}
    IDs = ['T{}{}'.format(tray, well) for tray in blocks.keys() for well in blocks[tray]]

    # Locations of input data and labels
    path1 = 'obsidian/datadump/{}_profiles.pickle'
    path2 = '/media/Elements/obsidian/diffraction_data/classes/{}_classifications.pickle' 

    print("Gathering data...")
    # Populate inputs and labels lists
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
    database = pd.DataFrame({'Path' : names, 'Data' : profiles, 'Class' : classes})
    
    pickle.dump(database, open('obsidian/datadump/database.pickle', 'wb'))

  def load_table(self, path):
    '''
    Load data from pre-pickled database and extract separate lists for inputs and classes

    :param str path: path of stored data file
    '''
    self.data_table = pickle_get(path)
    self.profiles = list(self.data_table['Data'])
    self.classes = list(self.data_table['Class'])
    assert len(self.profiles) == len(self.classes)

  def massage(self):
    '''
    Massage data into shape, prepare for model fitting, populate member variables
    '''
    assert self.data_table is not None, "Load data fist!" 

    # Reshape data to have dimensions (nsamples, ndata_points, 1)
    if len(np.stack(self.data_table['Data'].values).shape) < 3:
      self.data_table['Data'] = list(np.expand_dims(np.stack(self.data_table['Data'].values), 2))

    self.data_table = shuffle(self.data_table)
    
    # Split into train and test sets
    split_val = 0.8
    self.split = int(round(split_val*len(self.data_table)))

    self.train_data = self.data_table[:self.split]
    self.test_data = self.data_table[self.split:]
    
    # Extract training and test inputs and targets
    self.X_train, self.y_train = np.stack(self.train_data['Data'].values), np.stack(self.train_data['Class'].values)
    self.X_test, self.y_test = np.stack(self.test_data['Data'].values), np.stack(self.test_data['Class'].values)

  def print_summary(self):
    print("Data loaded!")
    print("Total number of samples: {0}\nBalance: class 1 - {1:.2f}%".format(len(self.profiles), (np.array(self.classes) == 1).sum()/len(self.classes)))
    print("Network to train:\n", self.model.summary())
    
  def build_model(self, nlayers=3, min_kern_size=3, max_kern_size=100, dropout=0.3, padding='same', loss='binary_crossentropy', loss_weight=0.5):
    '''
    Construct and compile a keras Sequential model according to spec

    :param int nlayers: number of 1D convolution layers (default 3)
    :param int min_kern_size: smallest kernel size (default 3)
    :param int max_kern_size: largest kernel size (default 100)
    :param float dropout: Dropout rate (default 0.3)
    :param str padding: padding mode for convolution layers (default 'same')
    :param str loss: loss function to use to determine weight updates
    :return: created and compiled keras model
    '''
    kernel_sizes = np.linspace(min_kern_size, max_kern_size, nlayers, dtype=int)
    nfilters = np.linspace(20, 40, nlayers, dtype=int)
    
    model = Sequential()

    # Input layer
    model.add(Conv1D(filters = nfilters[0], kernel_size=kernel_sizes[0].item(), padding=padding, activation='relu', input_shape=(2463, 1)))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))

    for i in range(1,nlayers):
      model.add(Conv1D(filters=nfilters[i], kernel_size=kernel_sizes[i].item(), padding=padding, activation='relu'))
      model.add(MaxPooling1D())
      model.add(Dropout(dropout))

    model.add(Flatten())
    
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    #model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #optimiser
    #adam = Adam(lr=0.01, decay=0.5)
  
    if loss == 'custom':
      loss = weighted_binary_crossentropy(weight=loss_weight)

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', precision])
    
    self.model = model
    return model

  def train_model(self, end=-1, epochs=30, batch_size=20):
    '''
    Shuffle and split data, then feed to model

    :param int epochs: number of training epochs (default 30)
    :param int batch_size: number of samples per weight update computation (default 20)
    :param int end: number of training samples (default use whole training set)
    '''
    assert self.model is not None, "Build model first!"
    
    self.massage()
    self.print_summary()

    self.history = self.model.fit(self.X_train[:end], self.y_train[:end], validation_data=(self.X_test, self.y_test), epochs=epochs, batch_size=batch_size).history
    
    self.model.save('obsidian/datadump/test-model.h5')
    pickle_put('obsidian/datadump/history.pickle', self.history)
    
    # Plot training history
    self.plot_train_performance(self.history)

  def model_from_save(self):
    print("WARNING: expect inaccurate performance readings when testing pre-trained models on seen data")

    self.model = load_model('obsidian/datadump/test-model.h5', custom_objects={'precision':precision})
    self.history = pickle_get('obsidian/datadump/history.pickle')

    # Plot training history
    self.plot_train_performance(self.history) 

  def test_model(self, batch_size=20):
    '''
    Test model after training. Display training stats and test results

    :param int batch_size:
    '''
    score = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
    print("Loss: {0:.2f}, Accuracy: {1:.2f}, Precision: {2:.2f}".format(score[0], score[1], score[2]))
    
    # Analyse performance
    predicted = self.model.predict_classes(self.X_test)
    probs = self.model.predict_proba(self.X_test)
    self.test_data['Prediction'] = probs

    cm = confusion_matrix(self.y_test, predicted)
    self.show_confusion(cm, [0,1])

    self.plot_test_results()

    '''
    indexed_test = indexed_data[self.split:]
    wrong_predictions = indexed_test[ predicted != indexed_test[:,-1]]
    '''

  def grid_search(self, batch_size=20, epochs=30, end=-1):
    '''
    Perform a grid search of parameter space to find optimal hyperparameter configuration
    '''
    model = KerasClassifier(build_fn=self.build_model, batch_size=batch_size, epochs=epochs, verbose=2)
    epochs = [35, 50]
    #dropout = [0.2, 0.3, 0.5]
    max_kern_size = [50, 100, 200, 400]
    min_kern_size = [3, 5, 10]
    #nlayers = [2, 3, 4]
    batch_size = [10, 20]
    #padding = ['valid', 'same']
    param_grid = dict(max_kern_size=max_kern_size, min_kern_size=min_kern_size, dropout=[0.3], nlayers=[4], padding=['valid'])
    #param_grid = dict(padding=padding, nlayers=nlayers, dropout=dropout)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=['accuracy', 'precision'], refit='precision')
    search_result = grid.fit(self.shuffled_data[:end,:-1], self.shuffled_data[:end,-1])
    
    results = pd.DataFrame(search_result.cv_results_)
    try:
      print(results)
    except Exception as e:
      print(e)
    try:
      pickle_put('obsidian/datadump/grid_search.pickle', results)
    except Exception as e:
      print(e)
      
  def plot_train_performance(self, history):
    '''
    Plot evolution of metrics such as accuracy and loss as a function of epoch number

    :param history: keras history object containing metric info from training
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
    
    ax1.plot(history['loss'], label = 'train', color = '#73ba71')
    ax1.plot(history['val_loss'], label = 'validation', color = '#5e9ec4')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    
    ax2.plot(history['acc'], label = 'train', color = '#73ba71')
    ax2.plot(history['val_acc'], label = 'validation', color = '#5e9ec4')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    
    ax3.plot(history['precision'], label = 'train', color = '#73ba71')
    ax3.plot(history['val_precision'], label = 'validation', color = '#5e9ec4')
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('precision')

    plt.legend(loc='lower right')
    plt.tight_layout()
  
  def plot_test_results(self):
    sort = self.test_data.sort_values('Prediction', ascending=False)

    x = np.arange(len(sort))
    preds = sort['Prediction'].values

    truex = np.where(sort['Class'] == np.rint(sort['Prediction']))[0]
    trues = sort.iloc[sort['Class'].values == np.rint(sort['Prediction'].values)]

    falsex = np.where(sort['Class'] != np.rint(sort['Prediction']))[0]
    falses = sort.iloc[sort['Class'].values != np.rint(sort['Prediction'].values)]
    
    plt.figure()
    plt.plot(x, preds, truex, trues['Class'].values, 'g.', falsex, falses['Class'].values, 'r.') 

  def show_confusion(self, cm, classes):
    '''
    Display confusion plot and stats

    :param array cm: calcualted confusion matrix
    :param classes: list of class labels
    '''

    # Stats
    [[tn, fp],[fn, tp]] = cm
    tpr = tp / (fn + tp) # True positive rate, Sensitivity, recall
    tnr = tn / (tn + fp) # True negative rate, Specificity
    ppv = tp / (tp + fp) # Positive predictive value, Precision
    npv = tn / (tn + fn) # Negative predictive value
    f1 = 2 * (ppv * tpr) / (ppv + tpr)
    stats = '{0:<20}{1:>10.2f}\n{2:<20}{3:>10.2f}\n{4:<20}{5:>10.2f}\n{6:<20}{7:>10.2f}\n{8:<20}{9:>10.2f}'.format('Sensitivity:', tpr, 'Specificity:',
    tnr, 'PPV (Precision):', ppv, 'NPV:', npv, 'F1 score:', f1)
    print(stats)

    # Plot
    plt.figure()
    try:
      plt.imshow(cm, interpolation='nearest', cmap='magma_r')
    except Exception as e:
      print(e)
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

def main(argv):

  build_kwargs = {}
  train_kwargs = {}
  mode='normal_testing'
  
  # Parse command line options
  try:
    opts, args = getopt.getopt(argv, 'n:b:e:d:p:l:w:o:', ['mode='])
  except getopt.GetoptError as e:
    print(e)
    print("convnet.py -n <num layers> -b <batch size> -e <num epochs> -d <size train data> --mode <default: 'normal_testing'>")
    sys.exit(2)
  if '-w' in opts and not 'custom' in args:
    print("Warning: providing loss weight meaningless if custom loss function not specified")
  for opt, arg in opts:
    if opt=='-n':
      build_kwargs['nlayers'] = int(arg)
    elif opt=='-p':
      build_kwargs['padding'] = arg
    elif opt=='-l':
      build_kwargs['loss'] = arg
    elif opt=='-w':
      build_kwargs['loss_weight'] = float(arg)
    elif opt=='-o':
      build_kwargs['dropout'] = float(arg)
    elif opt=='-b':
      train_kwargs['batch_size'] = int(arg)
    elif opt=='-e':
      train_kwargs['epochs'] = int(arg)
    elif opt=='-d':
      train_kwargs['end'] = int(arg)
    elif opt=='--mode':
      mode = arg
  
  ProteinClassifier.make_database()

  PC = ProteinClassifier()
  PC.load_table('obsidian/datadump/database.pickle')
  
  # Optional grid search run
  if mode=='grid':
    PC.massage()
    PC.grid_search(**train_kwargs)
  
  # Build and train model with default parameters except where 
  # specified otherwise
  elif mode=='saved':
    PC.massage()
    PC.model_from_save()
    PC.test_model()
  else:
    PC.build_model(**build_kwargs)
    PC.train_model(**train_kwargs)
    PC.test_model()
  
  plt.show()
  

if __name__ == '__main__':
  main(sys.argv[1:])
