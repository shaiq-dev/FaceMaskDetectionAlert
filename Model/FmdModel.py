import os
from os import path
import sys
import argparse
import datetime

import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam


def build_fmd_cnn(dataset, logs=False, save_plots=False):

    _images = []
    _labels = []
    IMG_ROWS = 112
    IMG_COLS = 112
    model = None

    _img_data = os.listdir(dataset)

    for ctg in _img_data:
        ctg_path = path.join(dataset, ctg)

        for img in os.listdir(ctg_path):
            if logs:
                print("[READING] {}".format(img))

            img = cv2.imread(path.join(ctg_path, img))

            # For better performance convert the img to grayscale
            try:
                img_greyscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _images.append(cv2.resize(img_greyscaled,
                                          (IMG_ROWS, IMG_COLS)))
                _labels.append(ctg)
            except Exception as _e:
                print(_e)

    print("[DONE] Images read successfully")

    _images = np.array(_images) / 255.0
    _images = np.reshape(_images, (_images.shape[0], IMG_ROWS, IMG_COLS, 1))

    # Hot encode labels because they are in textual form
    _lb_binarizer = LabelBinarizer()
    _labels = _lb_binarizer.fit_transform(_labels)
    _labels = to_categorical(_labels)
    _labels = np.array(_labels)

    (train_x, test_x, train_y, test_y) = train_test_split(_images,
                                                          _labels,
                                                          test_size=0.25,
                                                          random_state=0)

    # Strat Building the CNN
    # Model Params
    num_classes = 2
    batch_size = 32
    # Channeel_type: 1 - Greyscale, 3 - ColorImages
    channel_type = 1
    activation_type = "relu"

    model = Sequential()
    # First Layer
    model.add(
        Conv2D(64, (3, 3), input_shape=(IMG_ROWS, IMG_COLS, channel_type)))
    model.add(Activation(activation_type))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Second Layer
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation(activation_type))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 3rd Layer
    # Flatten and Dropout to stack the output convolutions above as well
    # as cater overfitting
    model.add(Flatten())
    model.add(Dropout(0.5))
    # 4th Layer
    # Softmax Classifier
    model.add(Dense(64, activation=activation_type))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())

    # Tranning Time

    # Epochs
    # Higher the value, greater the accuracy.
    # Higher Epochs will also result in more tranning time
    epochs = 50

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    fitted_fmd_model = model.fit(train_x,
                                 train_y,
                                 epochs=epochs,
                                 validation_split=0.25)

    if save_plots:
        from matplotlib import pyplot as plt

        # Tranning and Validation Loss
        plt.plot(fitted_fmd_model.history['loss'], 'r', label='Tranning Loss')
        plt.plot(fitted_fmd_model.history['val_loss'], label='Validation Loss')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.savefig("TranningValidationLoss.png")

        # Tranning and Validation Accuracy
        plt.plot(fitted_fmd_model.history['accuracy'],
                 'r',
                 label='Tranning Accuracy')
        plt.plot(fitted_fmd_model.history['val_accuracy'],
                 label='Validation Accuracy')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Accuracy Value')
        plt.legend()
        plt.savefig("TranningValidationAccuracy.png")

    model_name = "FaceMaskDetectionModel-{}.h5".format(datetime.datetime.now())
    fitted_fmd_model.save(model_name)
    print("[DONE] Saved Model {}".format(model_name))


if __name__ == '__main__':

    dataset_path = None
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        help="Add dataset for tranning the model",
                        required=True)

    parser.add_argument("--logs", "-l", default=False, help="Shows Logs")

    parser.add_argument(
        "--plot",
        "-p",
        default=False,
        help=
        "Saves Matplotlib plots of Tranning v/s Validation loss and Tranning v/s Validation accuracy"
    )

    args = parser.parse_args()

    logs = True if args.logs else False
    plot = True if args.plot else False

    if args.dataset:
        dataset_path = str(args.dataset)
        try:
            if os.path.exists(dataset_path):
                if len(os.listdir(dataset_path)) != 2:
                    raise ValueError(
                        "Two Labels expected in the dataset, found {}".format(
                            len(os.listdir(dataset_path))))
                else:
                    build_fmd_cnn(dataset_path, logs, plot)
            else:
                raise ValueError("Invalid Dataset Path")
        except Exception as _e:
            raise _e
    sys.exit(1)
