import pathlib
import json

import numpy as np
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.src.layers import Flatten, Dense, Dropout
from keras.src.utils import to_categorical

from app.images import ImageLoader
from app.preprocessing import OpenCvPreprocessing
from app.splitter import Splitter
from tensorflow.python.client import device_lib


PATH = str(pathlib.Path(__file__).parent.parent.resolve()) + "/data"
ANNOTATION_PATH = str(pathlib.Path(__file__).parent.parent.resolve()) + "/degrade-Degradant-annotations.json"

labels = []
filenames = []


print(device_lib.list_local_devices())

with open(ANNOTATION_PATH, "r") as file:
    data_json = json.load(file)
    labels =  [int(annotation["nameIdentifier"]) for annotation in data_json["annotations"]]
    filenames = [label["fileName"] for label in data_json["annotations"]]

image_loader = ImageLoader(PATH)
image_loader.load(filenames)

images = image_loader.images

open_cv_preprocessing = OpenCvPreprocessing(images)
open_cv_preprocessing.preprocess()

images = open_cv_preprocessing.images

splitter = Splitter(images, labels)
splitter.split(0.8)

splitter.train_y = to_categorical(splitter.train_y, 3)
splitter.test_y = to_categorical(splitter.test_y, 3)

splitter.train_x = np.array(splitter.train_x, dtype=np.float32)
splitter.test_x = np.array(splitter.test_x, dtype=np.float32)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print("train_x shape:", splitter.train_x.shape)
print("train_y shape:", splitter.train_y.shape)

model.fit(splitter.train_x, splitter.train_y, epochs=2, batch_size=32)
score = model.evaluate(splitter.test_x, splitter.test_y, verbose=2)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

yhat = model.predict(splitter.test_x[0])

print("coucou")

