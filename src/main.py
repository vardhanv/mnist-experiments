
import tensorflow as tf
from   tensorflow import keras
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense, Activation

from tensorflow.python.lib.io import file_io
import os
import gzip as gz
import struct as struct
import numpy as np
from matplotlib import pyplot as plt 



def mnistData (fname):
    # Lets open and display the files
    with gz.open(fname) as f:
      
        (mgk_no, no_of_imgs, nrows, ncols) = struct.unpack('>LLLL',f.read(16))
        print(mgk_no, no_of_imgs, nrows, ncols)
        d_arr = np.frombuffer(f.read(), 'b',count = -1,offset=0).reshape((no_of_imgs,-1))

        # Scale all values to be between 0 and 1
        d_arr = d_arr / 255
        return (d_arr, no_of_imgs, nrows, ncols)
   

def mnistLabels(fname):
    with gz.open(fname) as f:
        (mgk_no, no_of_labels) = struct.unpack('>LL',f.read(8))
        print ("No of labels = ", no_of_labels)
        l_arr = np.frombuffer(f.read(), 'b')
        
        label_vector = np.zeros((no_of_labels, 10))
        
        # set relevant output to 1
        label_vector [ np.arange(no_of_labels), l_arr] =  1
        
        return (label_vector, no_of_labels)

 
def createNNModel():
     model = keras.Sequential([
         # input layer 
         Dense(112,  Activation('relu'), input_shape=(784,)),
         
         # hidden layer - 1
         #Dense(32,  Activation('relu')),

         # output layer
         Dense(10, Activation('softmax'))])
     
     my_opti = keras.optimizers.RMSprop(lr=0.0001)
     compiledModel = model.compile(optimizer = my_opti,
                   loss    = 'categorical_crossentropy',
                   metrics = ['accuracy'])
     
     return model
     
     
     
 
(train_data, nimgs_td, nr_td, nc_td) = mnistData("../data/train-images-idx3-ubyte.gz")
#plt.imshow(train_data[0].reshape(nr_td,nc_td), cmap="binary")
#plt.show()

(train_labels, no_labels) = mnistLabels("../data/train-labels-idx1-ubyte.gz")

print (train_labels[0])

myModel = createNNModel()

batchSz = 60000
epochsNo = 10000

tb = keras.callbacks.TensorBoard(log_dir='../logs/h0-112-lr-0001-batch-60000-epochs-10000-with-scaling', histogram_freq=0, batch_size=batchSz,
                            write_graph=True, write_grads=True, write_images=True)


myModel.fit(train_data, train_labels, epochs = epochsNo, batch_size = batchSz , callbacks=[tb])
