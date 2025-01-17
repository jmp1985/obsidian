'''Module of Classes and methods for building, 
training and handling classification models
'''

# Basic packages
import sys, getopt, os
from glob import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# Machine learning
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
# Obsidian modules
from obsidian.utils.imgdisp import ImgDisp
from obsidian.utils.data_handling import pickle_get, pickle_put, new_database
from obsidian.learn.metrics import precision, weighted_binary_crossentropy

# Directories for saving and loading models and databases
models_dir = os.path.join(os.path.dirname(__file__), 'models')
data_dir = os.path.dirname(__file__)

class ProteinClassifier():
  '''
  Class for specifying, building, training and testing diffraction image classifiers

  :ivar data_table: database of all available images with image path, extracted data and class
  :ivar train_data:
  :ivar test_data:
  :ivar model: Keras model object
  :ivar history: epoch-wise training history, created when model trained
  :ivar split: number of data samples to be used for training (split fraction is 0.8)
  
  **Default parameters:**
    | number of layers: 6 
    | kernel sizes:     min 3, max 100
    | dropout:          0.3
    | padding:          same
    | number of epochs: 30
    | batch size:       20
    | loss weight:      0.5
  '''

  def __init__(self):
    '''When a ProteinClassifier object is instantiated, 
    data_table, model and history are set to None to 
    ensure that methods are called in the appropriate order.
    '''
    self.data_table = None
    self.model = None
    self.history = None
  
  @staticmethod
  def make_database(data_folder, labels_folder, name=''):
    '''Load data and classes out of relevant folders and store in a data frame along with
    file path for each image.

    :param str name: Database title. Database will be saved as "namedatabase.pickle"
    '''
    global data_dir
    save_path = os.path.join(data_dir, '{}database.pickle'.format(name))
    database = new_database(data_folder, label_folder, save_path=save_path)
    
  def load_database(self, path):
    '''Load data from pre-pickled database and extract separate lists for inputs and classes

    :param str path: path of stored data file
    '''
    self.data_table = pickle_get(path)

  def massage(self, split_val=0.8):
    '''Massage data into shape, prepare for model fitting, populate member variables

    :param float split_val: fraction of data to use for training, the remainder used for testing
    '''
    assert self.data_table is not None, "Load data fist!" 

    # Reshape data to have dimensions (nsamples, ndata_points, 1)
    if len(np.stack(self.data_table['Data'].values).shape) < 3:
      self.data_table['Data'] = list(np.expand_dims(np.stack(self.data_table['Data'].values), 2))

    self.data_table = shuffle(self.data_table)
    
    # Split into train and test sets
    self.split = int(round(split_val*len(self.data_table)))

    self.train_data = self.data_table[:self.split]
    self.test_data = self.data_table[self.split:]
    
    # Extract training and test inputs and targets
    if len(self.train_data)!=0:
      self.X_train, self.y_train = np.stack(self.train_data['Data'].values), np.stack(self.train_data['Class'].values)
    if len(self.test_data)!=0:
      self.X_test, self.y_test = np.stack(self.test_data['Data'].values), np.stack(self.test_data['Class'].values)

  def print_summary(self):
    '''Print summary of loaded samples and build model.
    '''
    print("Data loaded!")
    print("Total number of samples: {0}\nBalance: class 1 - {1:.2f}%".format(len(self.data_table), 
          (np.array(self.classes) == 1).sum()/len(self.classes)))
    print("Network to train:\n", self.model.summary())
    
  def build_model(self, nlayers=6, 
                  min_kern_size=3, max_kern_size=100, 
                  dropout=0.3, 
                  padding='same', 
                  custom_loss=False, loss_weight=0.5,
                  name='classifier_model'):
    '''Construct and compile a keras Sequential model according to spec

    :param int nlayers: number of 1D convolution layers (default 3)
    :param int min_kern_size: smallest kernel size (default 3)
    :param int max_kern_size: largest kernel size (default 100)
    :param float dropout: Dropout rate (default 0.3)
    :param str padding: padding mode for convolution layers (default 'same')
    :param bool custom_loss: if true, use weighted_binary_crossentropy
    :param float loss_weight: weighting of custom loss.  A value higher than 1 will bias the 
                              model towards positive predictions, a value lower than 1 will bias 
                              the model towards negative predictions.
    :return: created and compiled keras model
    '''
    kernel_sizes = np.linspace(min_kern_size, max_kern_size, nlayers, dtype=int)
    nfilters = np.linspace(30, 50, nlayers, dtype=int)
    
    model = Sequential()

    # Input layer
    model.add(Conv1D(filters = nfilters[0], kernel_size=kernel_sizes[0].item(), 
                     padding=padding, activation='relu', input_shape=(2000, 1)))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout))

    for i in range(1,nlayers):
      model.add(Conv1D(filters=nfilters[i], kernel_size=kernel_sizes[i].item(), 
                       padding=padding, activation='relu'))
      model.add(MaxPooling1D())
      model.add(Dropout(dropout))

    model.add(Flatten())
    
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
  
    #if loss == 'custom':
    loss = weighted_binary_crossentropy(weight=loss_weight) if custom_loss else 'binary_crossentropy'

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', precision])
    
    self.model = model
    
    global models_dir

    with open(os.path.join(models_dir, '{}.txt'.format(name)), 'w') as f:
      f.write("loss "+("weighted_binary_crossentropy(weight=loss_weight)\n" if custom_loss else "binary_crossentropy\n"))
      f.write("loss_weight "+str(loss_weight)+"\n")

    return model

  def train_model(self, epochs=30, batch_size=20, 
                  name='classifier_model', 
                  update=False, end=-1):
    '''Shuffle and split data, then feed to model

    :param int epochs: number of training epochs (default 30)
    :param int batch_size: number of samples per weight update computation (default 20)
    :param int end: number of training samples (default use whole training set)
    '''
    assert self.model is not None, "Build model first!"
    if update:
      assert self.history is not None, "Load pre-trained model to use update option"
    
    self.massage()
    self.print_summary()

    history = self.model.fit(self.X_train[:end], self.y_train[:end], 
                             validation_data=(self.X_test, self.y_test), 
                             epochs=epochs, batch_size=batch_size).history
    
    if update:
      for key in self.history.keys():
        self.history[key].extend(history[key])
    else:
      self.history = history

    self.model.save(os.path.join(models_dir, '{}.h5'.format(name)))
    pickle_put(os.path.join(models_dir, '{}_history.pickle'.format(name)), self.history)

    # Plot training history
    self.plot_train_performance(self.history)

  def model_from_save(self, name='classifier_model'):
    '''Load a prebuilt model from file as instance model.
    Model must have an associated .txt file recording its loss
    function.

    :param str name: name of saved model to be loaded
    '''
    print("WARNING: expect inaccurate performance readings if testing pre-trained models on seen data")

    global models_dir

    with open(os.path.join(models_dir, '{}.txt'.format(name))) as f:
      params = {line.split(' ')[0] : line.split(' ')[1] for line in f}

    loss_weight = eval(params['loss_weight'])
    try:
      loss = eval(params['loss'])
      custom_objects = {'precision':precision, 'weighted_loss':loss}
    except NameError:
      custom_objects = {'precision':precision}

    self.model = load_model(os.path.join(models_dir, '{}.h5'.format(name)), 
                            custom_objects=custom_objects)
    self.history = pickle_get(os.path.join(models_dir, '{}_history.pickle'.format(name)))

    # Plot training history
    self.plot_train_performance(self.history) 

  def test_model(self, batch_size=20):
    '''Test model after training. Display training stats and test results.

    :param int batch_size:
    '''

    # Score model on test data
    score = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
    print("Loss: {0:.2f}, Accuracy: {1:.2f}, Precision: {2:.2f}".format(score[0], score[1], score[2]))
    
    # Analyse performance, add predictions to loaded database
    predicted = self.model.predict_classes(self.X_test)
    probs = self.model.predict_proba(self.X_test)
    self.test_data['Prediction'] = probs
    
    # Display results graphically
    cm = confusion_matrix(self.y_test, predicted)
    self.show_confusion(cm, [0,1])

    self.plot_test_results()

  def plot_train_performance(self, history):
    '''Plot evolution of metrics such as accuracy and loss as a function of epoch number

    :param history: keras history object containing metric info from training
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1)
    fig.dpi = 300 
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
    '''Produce a plot of model predictions against true labels
    '''

    # Sort samples by predicted probability
    sort = self.test_data.sort_values('Prediction', ascending=False)

    x = np.arange(len(sort))
    preds = sort['Prediction'].values

    # Determine indices of true and false predictions
    truex = np.where(sort['Class'] == np.rint(sort['Prediction']))[0]
    trues = sort.iloc[sort['Class'].values == np.rint(sort['Prediction'].values)]

    falsex = np.where(sort['Class'] != np.rint(sort['Prediction']))[0]
    falses = sort.iloc[sort['Class'].values != np.rint(sort['Prediction'].values)]
    
    # Construct figure
    plt.figure(dpi=300)
    
    plt.plot(x, preds, linewidth=0.5) # Predictions
    plt.plot(truex, trues['Class'].values, 'g.', markersize=1.5) # True labels, correct
    plt.plot(falsex, falses['Class'].values, 'r.', markersize=1.5) # True labels, incorrect
    
    plt.xticks([0, len(sort)])
    plt.yticks([0, 0.5, 1])
    
    plt.xlabel('Sample number')
    plt.ylabel('Probability')

  def show_confusion(self, cm, classes):
    '''Display confusion plot and stats

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
    stats = '{0:<20}{1:>10.2f}\n{2:<20}{3:>10.2f}\n{4:<20}{5:>10.2f}\n{6:<20}{7:>10.2f}\n{8:<20}{9:>10.2f}'.format('Sensitivity:', tpr, 'Specificity:', tnr, 'PPV (Precision):', ppv, 'NPV:', npv, 'F1 score:', f1)
    print(stats)
    
    # Construct figure 
    plt.figure(dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap='magma_r')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Matrix numerical values
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', 
               color = 'black' if cm[i,j]<thresh else 'white')

    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    
  def show_wrongs(self, n=10):
    '''Display wongly classified images 
    together with their true class and predicted class probabilities.
    Print wrongly predicted images in tabular form.

    :param int n: number of wrongs to display 
    '''
    wrongs = self.test_data.iloc[self.test_data['Class'].values != np.rint(self.test_data['Prediction'].values)]
    num = len(wrongs)
    w = ImgDisp([np.load(f) for f in wrongs['Path'].values][:n])

    fig1, ax1 = w.disp()
    fig1.subplots_adjust(wspace=0.02, hspace=0.02)

    try:
      for i in range(n):
        ax1.flat[i].set_title((str(wrongs.iloc[i]['Class'])+' '+str(wrongs.iloc[i]['Prediction'])))
    except Exception as e:
      print("Couldn't label plots: \n", e)
    
    pd.set_option('display.max_colwidth', 80)
    print(wrongs[['Path','Class','Prediction']])

def main(argv):
  
  global data_dir

  build_kwargs = {}
  train_kwargs = {}
  mode = 'normal_testing'
  remake = False
  name = 'classifier_model'

  # Parse command line options
  try:
    opts, args = getopt.getopt(argv, 'n:b:e:d:p:w:o:', ['mode=', 'remake', 'name=', 'custom_loss', 'data='])
  except getopt.GetoptError as e:
    print(e)
    print("convnet.py \
          -n <num layers> \
          -b <batch size> \
          -e <num epochs> \
          -d <size train data> \
          --mode <default: 'normal_testing'> \
          --name <model name (if name pre-exists will overwrite old model)> \
          --remake to rebuild database")
                    
    sys.exit(2)
  for opt, arg in opts:
    if opt=='-n':
      build_kwargs['nlayers'] = int(arg)
    elif opt=='-p':
      build_kwargs['padding'] = arg
    elif opt=='--custom_loss':
      build_kwargs['custom_loss'] = True
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
    elif opt=='--remake':
      remake = True
    elif opt=='--name':
      train_kwargs['name'] = arg
      build_kwargs['name'] = arg
      name = arg
    elif opt=='--data':
      data_dir = arg 
  
  if remake:
    ProteinClassifier.make_database()

  PC = ProteinClassifier()

  avail = glob(os.path.join(data_dir, '*database.pickle'))
  if os.path.isfile(data_dir):
    data_path = data_dir
  elif len(avail) == 1:
    data_path = avail[0]
  else:
    data_path = input("Available databases:\n\t"+"\n\t".join(name for name in avail)+"\nSelect from above: ")

  PC.load_database(data_path)
   
  # Build and train model with default parameters except where 
  # specified otherwise
  if mode=='saved':
    PC.massage(split_val=0)
    PC.model_from_save(name=name)
    PC.test_model()
    PC.show_wrongs()
  elif mode=='update':
    PC.model_from_save(name=name) 
    PC.train_model(update=True, **train_kwargs)
    PC.test_model()
  else:
    go_ahead = True
    if os.path.isfile(os.path.join(models_dir, '{}.h5'.format(name))):
       go_ahead = (input("'{}' is an existing model. \
       Press 'y' to overwrite, any other key to cancel. ".format(name)) == 'y')
    if go_ahead:
      PC.build_model(**build_kwargs)
      PC.train_model(**train_kwargs)
      PC.test_model()
      #PC.show_wrongs()
    else:
      sys.exit(2)
      
  plt.show()
  

if __name__ == '__main__':
  main(sys.argv[1:])
