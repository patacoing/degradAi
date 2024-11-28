from abc import ABC, abstractmethod

import numpy as np
from keras import Sequential
from keras.src.applications.vgg16 import VGG16
from keras.src.layers import Flatten, Dense, Dropout
from keras.src.utils import to_categorical


class ITraining(ABC):
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = np.array(train_x)
        self.train_y = to_categorical(np.array(train_y, dtype=int), num_classes=3)
        self.test_x = np.array(test_x)
        self.test_y = to_categorical(np.array(test_y, dtype=int), num_classes=3)

    @abstractmethod
    def train(self, epochs: int, batch_size: int) -> Sequential:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: Sequential) -> tuple[float, float]:
        raise NotImplementedError


class Training(ITraining):
    def train(self, epochs: int, batch_size: int) -> Sequential:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        base_model.trainable = False

        model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        model.fit(
            self.train_x,
            self.train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.test_x, self.test_y)
        )

        return model

    def evaluate(self, model: Sequential) -> tuple[float, float]:
        return model.evaluate(self.test_x, self.test_y)