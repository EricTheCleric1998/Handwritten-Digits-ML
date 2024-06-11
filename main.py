# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# # mnist = tf.keras.datasets.mnist
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# #
# # x_train = tf.keras.utils.normalize(x_train, axis=1)
# # x_test = tf.keras.utils.normalize(x_test, axis=1)
# #
# # model = tf.keras.models.Sequential()
# # model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dense(128, activation='relu'))
# # model.add(tf.keras.layers.Dense(10, activation='softmax'))
# #
# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #
# # model.fit(x_train, y_train, epochs=10)
# #
# # model.save('handwritten.keras')
#
# model = tf.keras.models.load_model('handwritten.keras')
#
# image_number = 1
#
# while os.path.isfile(f"digits/digit{image_number}.png"):
#     try:
#         img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         print(f"This digit is probably a {np.argmax(prediction)}")
#         plt.imshow(img[0], cmap=plt.cm.binary)
#     except:
#         print("Error!")
#     finally:
#         image_number += 1

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train", action='store_true',
                    help="Train a new model and then test it.")
parser.add_argument("--cnn", action='store_true',
                    help="Train with cnn model, otherwise train fully connected.")
options = parser.parse_args()

if options.train or options.cnn:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()

if options.train:
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

elif options.cnn:
    # Build the CNN model
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

if options.train or options.cnn:
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')

image_number = 1

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)

        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except:
        print("Error!")
    finally:
        image_number += 1