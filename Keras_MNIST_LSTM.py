import keras
import numpy as np
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam

learning_rate = 0.001
training_iters = 20
batch_size = 128
display_step = 10

n_input = 28
n_step = 28
n_hidden = 128
n_classes = 10

# the data, split between train and test sets
train_data = np.load('train_data.npy') 
test_data = np.load('test_data.npy') 

train_x = [i[0] for i in train_data]
print(np.shape(train_x))

train_y = [i[1] for i in train_data]
print(np.shape(train_y))

test_x = [i[0] for i in test_data]
print(np.shape(test_x))

test_y = [i[1] for i in test_data]
print(np.shape(test_y))

train_x = np.array(train_x, dtype = np.float32)
train_y = np.array(train_y, dtype = np.float32)
test_x = np.array(test_x, dtype = np.float32)
test_y = np.array(test_y, dtype = np.float32)

model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, n_step, n_input), unroll=True))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
adam = Adam(lr=learning_rate)
model.summary()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size = batch_size, epochs = training_iters, verbose = 1)
scores = model.evaluate(test_x, test_y, verbose=0)

print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])