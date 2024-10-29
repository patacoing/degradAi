import numpy as np
from abc import ABC
from sklearn.model_selection import train_test_split


class ISplitter(ABC):
    def split(self, ratio: float):
        ...

class Splitter(ISplitter):
    def __init__(self, images: list[np.array]):
        self.images = images
        self.train = []
        self.test = []


    def split(self, ratio: float):
         self.train, self.test = train_test_split(self.images, train_size=ratio)

