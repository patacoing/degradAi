from abc import ABC, abstractmethod
import cv2
import numpy as np


class IPreprocessing(ABC):
    @abstractmethod
    def resize(self):
        raise NotImplementedError

    @abstractmethod
    def grayscale(self):
        raise NotImplementedError

    @abstractmethod
    def normalize(self):
        raise NotImplementedError

    def preprocess(self) -> np.array:
        self.resize()
        self.grayscale()
        self.normalize()

        return self.images


class OpenCvPreprocessing(IPreprocessing):
    def __init__(self, images: list[np.array]):
        self.images = images

    def resize(self):
        tmp = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in self.images]
        self.images = np.array([cv2.resize(image, (300, 300)) for image in tmp])

    def grayscale(self):
        images_pregray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in self.images]
        self.images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_pregray])

    def normalize(self):
        self.images = self.images / 255