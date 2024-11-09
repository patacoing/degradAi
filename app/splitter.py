import numpy as np
from abc import ABC, abstractmethod
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split


class ISplitter(ABC):
    @abstractmethod
    def split(self, ratio: float):
        raise NotImplementedError

    @abstractmethod
    def to_categorical(self, num_classes: int):
        raise NotImplementedError


class Splitter(ISplitter):
    def __init__(self, images: list[np.array], labels: list[str]):
        self.images = images
        self.labels = labels

    def split(self, ratio: float):
         self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.images, self.labels, train_size=ratio)

    def to_categorical(self, num_classes: int):
        self.train_y = to_categorical(self.train_y, num_classes)
        self.test_y = to_categorical(self.test_y, num_classes)

