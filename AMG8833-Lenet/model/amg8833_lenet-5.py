from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint

import numpy as np
import os
import sys
nnscript = os.path.abspath('../../scripts')
sys.path.append(nnscript)

from nnom_utils import *
from utils import *

model_name = 'AMG8833_trained_model'
save_dir = os.path.join(os.getcwd(), 'saved_models')

def image_to_cfile(data, label, size, file='image.h'):
    # test
    with open(file, 'w') as f:
        num_of_image = size
        f.write('#include "nnom.h"\n\n')
        for i in range(num_of_image):
            selected = np.random.randint(0, 50) # select 10 out of 50.
            f.write('#define IMG%d {'% (i))
            np.round(data[selected]).flatten().tofile(f, sep=", ", format="%d") # convert 0~1 to 0~127
            f.write('} \n')
            f.write('#define IMG%d_LABLE'% (i))
            f.write(' %d \n \n' % label[selected])
        f.write('#define TOTAL_IMAGE %d \n \n'%(num_of_image))

        f.write('static const int8_t img[%d][%d] = {' % (num_of_image, data[0].flatten().shape[0]))
        f.write('IMG0')
        for i in range(num_of_image -1):
            f.write(',IMG%d'%(i+1))
        f.write('};\n\n')

        f.write('static const int8_t label[%d] = {' % (num_of_image))
        f.write('IMG0_LABLE')
        for i in range(num_of_image -1):
            f.write(',IMG%d_LABLE'%(i+1))
        f.write('};\n\n')


def train(x_train, y_train, x_test, y_test, num_classes, batch_size= 64, epochs = 100):

    inputs = Input(shape=x_train.shape[1:])
    x = UpSampling2D()(inputs)
    x = UpSampling2D()(x)

    x = Conv2D(6, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D((2,2),strides=(2,2), padding="same")(x)

    x = Conv2D(16 ,kernel_size=(3,3), strides=(1,1), padding="same")(x)
    x = ReLU()(x)
    x = MaxPool2D((2,2),strides=(2,2), padding="same")(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = ReLU()(x)

    # x = Dense(50)(x)
    # x = Dropout(0.5)(x)
    # x = ReLU()(x)

    x = Dense(num_classes)(x)
    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.summary()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save best
    checkpoint = ModelCheckpoint('./saved_models/AMG8833_trained_model.h5',
        monitor='val_acc', verbose=2, save_best_only='True',  mode='auto', period=1)
    callback_lists = [checkpoint]

    history =  model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=callback_lists)

    model.save('./saved_models/AMG8833_trained_model.h5')

    del model
    K.clear_session()

    return history


if __name__ == "__main__":
    epochs = 200
    num_classes = 4

    # The data, split between train and test sets:
    (x_train, y_train_num), (x_test, y_test_num) = amg8833_load_data()

    print(x_train.shape[0], 'train image samples')
    print(x_test.shape[0], 'test image samples')
    print(y_train_num.shape[0], 'train label samples')
    print(y_test_num.shape[0], 'test label samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train_num, num_classes)
    y_test = keras.utils.to_categorical(y_test_num, num_classes)

    # reshape to 4 d becaue we build for 4d?
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    print('x_train shape:', x_train.shape)
    print('y_test shape:', y_test.shape)

    # quantize the range to 0~255 -> 0~1
    x_test = x_test/255.0
    x_train = x_train/255.0
    print("data range", x_test.min(), x_test.max())

    # select a few image and write them to image.h
    image_to_cfile(x_test*127, y_test_num, 30, file='image.h')

    # train model, save the best accuracy model
    history = train(x_train, y_train, x_test, y_test, num_classes, batch_size=64, epochs=epochs)

    # reload best model
    model = load_model('./saved_models/AMG8833_trained_model.h5')

    # evaluate
    evaluate_model(model, x_test, y_test)

    # save weight
    generate_model(model, np.vstack((x_train, x_test)), format='hwc', name="weights.h")

    # plot
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(range(0, epochs), acc, color='red', label='Training acc')
    plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()





















