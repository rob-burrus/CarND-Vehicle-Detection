import numpy as np
import tensorflow as tf
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
import cv2
import os
import pickle
import glob
from datetime import datetime

def pickle_data():
    basedir = 'vehicles/'
    image_types = os.listdir(basedir)
    cars = []
    for imtype in image_types:
        cars.extend(glob.glob(basedir+imtype+'/*'))
    print ('Number of Vehicle Images found: ', len(cars))
    with open('cars.txt', 'w') as f:
        for fn in cars:
            f.write(fn + '\n')

    basedir = 'non-vehicles/'
    image_types = os.listdir(basedir)
    notcars = []
    for imtype in image_types:
        notcars.extend(glob.glob(basedir+imtype+'/*'))
    print ('Number of Non-Vehicle Images found: ', len(notcars))
    with open('non-cars.txt', 'w') as f:
        for fn in notcars:
            f.write(fn + '\n')

    files = cars + notcars
    y = np.concatenate((np.ones(len(cars)), np.zeros(len(notcars))))

    files, y = shuffle(files, y)

    X_train, X_test, y_train, y_test = train_test_split(files, y, test_size=0.2, random_state=19)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=19)

    data = {'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test}

    pickle.dump(data, open('data.p', 'wb'))


def load_data():
    if not os.path.isfile('data.p'):
        pickle_data()

    with open('data.p', mode='rb') as f:
        data = pickle.load(f)
        X_train = data['X_train']
        X_test = data['X_test']
        X_val = data['X_val']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']

    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)


def get_images(img_paths, labels, batch_size):
    #print("Here")
    imgs = np.empty([batch_size, 64, 64, 3])
    for i,path in enumerate(img_paths):
        #print("Image path:{}".format(path))
        imgs[i] = cv2.imread(path)

    return imgs, labels

def generate(x, y, batch_size):
    size = len(x)
    while True:
        rng = np.random.choice(size, batch_size)
        x_batch, y_batch = get_images(x[rng], y[rng], batch_size)
        yield x_batch, y_batch

def get_model():
    inputShape = (64, 64, 3)
    model = Sequential()
    #model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(64, 64, 3)))
    #model.add(Convolution2D(16, 3, 3, input_shape=(64,64,3), activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Convolution2D(32, 3, 3, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(8, 8)))
    #model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(50))
    #model.add(Dropout(0.5))
    #model.add(Dense(1))
    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda x: x / 255., input_shape=inputShape))
    # Block 0
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='cv0', input_shape=inputShape, padding="same"))
    model.add(Dropout(0.5))

    # Block 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='cv1', padding="same"))
    model.add(Dropout(0.5))

    # block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    # binary 'classifier'
    model.add(Conv2D(filters=1, kernel_size=(8, 8), name='fcn', activation="sigmoid"))
    return model

def generator(samples, batchSize=32, useFlips=False, resize=False):
    """
    Generator to supply batches of sample images and labels
    :param samples: list of sample images file names
    :param batchSize:
    :param useFlips: adds horizontal flips if True (effectively inflates training set by a factor of 2)
    :param resize: Halves images widths and heights if True
    :return: batch of images and labels
    """
    samplesCount = len(samples)

    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, samplesCount, batchSize):
            batchSamples = samples[offset:offset + batchSize]

            xTrain = []
            yTrain = []
            for batchSample in batchSamples:
                y = float(batchSample[1])

                fileName = batchSample[0]

                image = aux.rgbImage(fileName, resize=resize)

                xTrain.append(image)
                yTrain.append(y)

                if useFlips:
                    flipImg = aux.flipImage(image)
                    xTrain.append(flipImg)
                    yTrain.append(y)

            xTrain = np.array(xTrain)
            yTrain = np.expand_dims(yTrain, axis=1)

            yield shuffle(xTrain, yTrain)  # Since we added flips, better shuffle again


def createSamples(x, y):
    """
    Returns a list of tuples (x, y)
    :param x:
    :param y:
    :return:
    """
    assert len(x) == len(y)

    return [(x[i], y[i]) for i in range(len(x))]



def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    #print("data loaded")

    sourceModel = get_model()

    x = sourceModel.output
    x = Flatten()(x)
    model = Model(inputs=sourceModel.input, outputs=x)

    #print(model.summary())

    # Train the Model
    model.compile('adam', 'mse', metrics=['accuracy'])

    batch_size = 1

    #train_gen = generate(X_train, y_train, batch_size)
    #valid_gen = generate(X_val, y_val, batch_size)

    trainGen = generator(samples=trainSamples, useFlips=useFlips)
    validGen = generator(samples=validationSamples, useFlips=useFlips)

    weights_file = 'model_{}.h5'.format(datetime.now())

    checkpointer = ModelCheckpoint(filepath=weightsFile,
                                   monitor='val_acc', verbose=0, save_best_only=True)

    history = model.fit_generator(train_gen,
                                  nb_epoch=4,
                                  samples_per_epoch=(len(X_train)//batch_size)*batch_size,
                                  validation_data=valid_gen,
                                  nb_val_samples=(len(X_val)//batch_size)*batch_size,
                                  verbose=1)

    # Save the Model
    model.save(weights_file)

    print('Evaluating accuracy on test set.')

    accuracy = model.evaluate_generator(generator=generate(X_test, y_test, batch_size), val_samples=(len(X_test)//batch_size)*batch_size)

    print('test accuracy: ', accuracy)


if __name__ == '__main__':
    main()
