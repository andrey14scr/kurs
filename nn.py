import keras
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 400
num_classes = 10
epochs = 1
iterations_count = 10;

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

sum = 0
for j in range(iterations_count):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.1,
                                                             beta_1=0.05,
                                                             beta_2=0.05,
                                                             epsilon=1e-07,
                                                             amsgrad=False,
                                                             name='Adam', ),
                  metrics=['accuracy'])

    hist = model.fit(train_data,
                     train_labels_cat,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(test_data, test_labels_cat))

    #print("The model has successfully trained")
    score = model.evaluate(test_data, test_labels_cat, verbose=0)
    sum += score[1]
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

print('Average accuracy:', sum / 10.0)
#model.save('mnist.h5')
#print("Saving the model as mnist.h5")
