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
learning_rate = 0.01
alpha = 0.05
beta = 0.05
# Dropout parameters:
dropout_step = 0.1
# NN parameters:
hidden_layer_size = 40
iterations_count = 20
epochs_list = [2]
batch_size_list = [400]

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

results = []
dropout = 0
for epochs in epochs_list:
    for batch_size in batch_size_list:
        f = open("nn.txt", "a")
        f.write(f'\n    Epochs: {epochs}, Batch size: {batch_size}, Iterations: {iterations_count} ({hidden_layer_size})\n\n')
        f.close()
        while(dropout < 0.91):
            tmp = []
            times = []
            for j in range(iterations_count):
                model = Sequential()
                model.add(Dense(hidden_layer_size, activation='relu', input_shape=(size,)))
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
                times.append((toc - tic) * 1000)
                score = model.evaluate(test_data, test_labels_cat, verbose=0)
                tmp.append(score[1])
            results.append({
                "dp": round(dropout * 100, 1),
                "acc": ave(tmp) * 100,
                "time": ave(times)
            })
            f = open("nn.txt", "a")
            f.write(f'Dropout: {results[-1]["dp"]:4.{1}f}%, Accuracy: {results[-1]["acc"]:.{3}f}%, Min: {(min(tmp)*100):.{2}f}%, Max: {(max(tmp)*100):.{2}f}%, Time: {results[-1]["time"]:.{3}f}ms\n')
            f.close()
            dropout += dropout_step

f = open("nn.txt", "a")
f.write('\nFor excel:\n')
for r in results:
    f.write(f'{r["dp"]:4.{1}f}\t{r["acc"]:.{3}f}\t{r["time"]:.{3}f}\n')
f.close()