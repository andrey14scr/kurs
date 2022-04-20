import keras
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import time

NUM_CLASSES = 10
EPSILON = 1e-07

def ave(list):
    return sum(list) / len(list)

def getLayers(input_size):
    return [
        Dense(25, activation='relu', input_shape=(input_size,))
    ]

# Adam parameters:
learning_rate = 0.1
alpha = 0.05
beta = 0.05
# Dropout parameters:
dropout_step = 0.1
# NN parameters:
iterations_count = 12
epochs_list = [1]
batch_size_list = [1000]

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
size = image_size*image_size

train_data = x_train.reshape(x_train.shape[0], size)
test_data = x_test.reshape(x_test.shape[0], size)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255.0
test_data /= 255.0

train_labels_cat = keras.utils.to_categorical(y_train, NUM_CLASSES)
test_labels_cat = keras.utils.to_categorical(y_test, NUM_CLASSES)

dropout = 0
for epochs in epochs_list:
    for batch_size in batch_size_list:
        f = open("nn.txt", "a")
        f.write(f'\n    Epochs: {epochs}, Batch size: {batch_size}, Iterations: {iterations_count}\n\n')
        f.close()
        while(dropout < 0.91):
            res = []
            times = []
            for j in range(iterations_count):
                model = Sequential()
                for layer in getLayers(size):
                    model.add(layer)
                    model.add(Dropout(dropout))
                model.add(Dense(NUM_CLASSES, activation='softmax'))

                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate,
                                                                         beta_1=alpha,
                                                                         beta_2=beta,
                                                                         epsilon=EPSILON,
                                                                         amsgrad=False,
                                                                         name='Adam', ),
                              metrics=['accuracy'])

                tic = time.perf_counter()
                hist = model.fit(train_data,
                                 train_labels_cat,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=0,
                                 validation_data=(test_data, test_labels_cat))
                toc = time.perf_counter()
                times.append(toc - tic)
                score = model.evaluate(test_data, test_labels_cat, verbose=0)
                res.append(score[1])
            f = open("nn.txt", "a")
            f.write(f'Dropout: {(dropout * 100):4.{1}f}%, Accuracy: {ave(res):.{6}f}, Min: {min(res):.{6}f}, Max: {max(res):.{6}f} Time: {ave(times):.{6}f}\n')
            f.close()
            dropout += dropout_step

#model.save('mnist.h5')