import numpy as np
from abc import ABC
from sklearn.model_selection import train_test_split


class ISplitter(ABC):
    def split(self, ratio: float):
        ...

class Splitter(ISplitter):
    def __init__(self, images: list[np.array], labels: list[str]):
        self.images = images
        self.labels = labels


    def split(self, ratio: float):
         self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.images, self.labels, train_size=ratio)

