import keras
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import time

def ave(list):
    return sum(list) / len(list)

num_classes = 10
iterations_count = 10
dropout_iterations = 10
hidden_layer_size = 25
input_layer_size = 28 * 28
learning_rate = 0.1
alpha = 0.05
beta = 0.05
epochs = 2
batch_size = 400

(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]

train_data = x_train.reshape(x_train.shape[0], image_size*image_size)
test_data = x_test.reshape(x_test.shape[0], image_size*image_size)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255.0
test_data /= 255.0

train_labels_cat = keras.utils.to_categorical(y_train, num_classes)
test_labels_cat = keras.utils.to_categorical(y_test, num_classes)

"""
for e in [1, 2, 3, 4]:
    for bs in [200, 400, 600]:
        f = open("nn.txt", "a")
        f.write(f'\n    Epochs: {e}, Batch size: {bs}\n\n')
        f.close()
        for i in range(dropout_iterations):
            res = []
            times = []
            dropout = 0.10 * i
            for j in range(iterations_count):
                tic = time.perf_counter()
                model = Sequential()
                model.add(Dense(hidden_layer_size, activation='relu', input_shape=(input_layer_size,)))
                model.add(Dropout(dropout))
                model.add(Dense(num_classes, activation='softmax'))

                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate,
                                                                         beta_1=alpha,
                                                                         beta_2=beta,
                                                                         epsilon=1e-07,
                                                                         amsgrad=False,
                                                                         name='Adam', ),
                              metrics=['accuracy'])

                hist = model.fit(train_data,
                                 train_labels_cat,
                                 batch_size=bs,
                                 epochs=e,
                                 verbose=1,
                                 validation_data=(test_data, test_labels_cat))
                toc = time.perf_counter()
                times.append(toc - tic)
                score = model.evaluate(test_data, test_labels_cat, verbose=0)
                res.append(score[1])
            f = open("nn.txt", "a")
            #f.write(f'Dropout: {(dropout*100):4.{1}f}%, Accuracy: {(sum(res) / iterations_count):.{7}f}, Min: {min(res):.{7}f}, Max: {max(res):.{7}f}\n')
            f.write(f'Dropout: {(dropout * 100):4.{1}f}%, Accuracy: {(sum(res) / iterations_count):.{7}f}, Min: {min(res):.{7}f}, Max: {max(res):.{7}f} Time: {times}\n')
            f.close()
"""

for i in range(dropout_iterations):
    res = []
    times = []
    dropout = 0.10 * i
    for j in range(iterations_count):
        model = Sequential()
        model.add(Dense(hidden_layer_size, activation='relu', input_shape=(input_layer_size,)))
        model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate,
                                                                 beta_1=alpha,
                                                                 beta_2=beta,
                                                                 epsilon=1e-07,
                                                                 amsgrad=False,
                                                                 name='Adam', ),
                      metrics=['accuracy'])

        tic = time.perf_counter()
        hist = model.fit(train_data,
                         train_labels_cat,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(test_data, test_labels_cat))
        toc = time.perf_counter()
        times.append(toc - tic)
        score = model.evaluate(test_data, test_labels_cat, verbose=0)
        res.append(score[1])
    f = open("nn.txt", "a")
    f.write(f'Dropout: {(dropout * 100):4.{1}f}%, Accuracy: {ave(res):.{6}f}, Min: {min(res):.{6}f}, Max: {max(res):.{6}f} Time: {ave(times):.{6}f}\n')
    f.close()

#model.save('mnist.h5')
#print("Saving the model as mnist.h5")