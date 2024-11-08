import glob
from abc import ABC
import cv2
import numpy as np

class ILoader(ABC):
    @classmethod
    def load(cls, file_path: str):
        ...

class OpenCvLoader:
    @classmethod
    def load(cls, file_path: str) -> np.array:
        return cv2.imread(file_path)


class IImageLoader(ABC):
    def __init__(self, path: str, loader: ILoader = OpenCvLoader()):
        self.path = path
        self.images = []
        self.loader = loader

    def load(self, filenames: list[str]):
        ...


class ImageLoader(IImageLoader):
    def load(self, filenames: list[str]):
        self.images = [self.loader.load(f"{self.path}/{filename}") for filename in filenames]
