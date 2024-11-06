from abc import ABC
import cv2
import numpy as np


class IPreprocessing(ABC):
    def resize(self):
        ...

    def grayscale(self):
        ...

    def normalize(self):
        ...

    def preprocess(self):
        self.resize()
        self.grayscale()
        self.normalize()


class OpenCvPreprocessing(IPreprocessing):
    def __init__(self, images: list[np.array]):
        self.images = images

    def resize(self):
        tmp_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in self.images]
        self.images = [cv2.resize(image, (300, 3000)) for image in tmp_images]

    def grayscale(self):
        images_pregray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in self.images]
        self.images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_pregray]

    def normalize(self):
        self.images = [image / 255.0 for image in self.images]