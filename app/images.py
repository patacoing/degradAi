import glob
from abc import ABC
import cv2
import numpy as np


class IFindFiles(ABC):
    @classmethod
    def find(cls, path: str) -> list[str]:
        ...

class FindFiles(IFindFiles):
    @classmethod
    def find(cls, path: str) -> list[str]:
        return glob.glob(f"{path}/**")


class ILoader(ABC):
    @classmethod
    def load(cls, file_path: str):
        ...

class OpenCvLoader:
    @classmethod
    def load(cls, file_path: str) -> np.array:
        return cv2.imread(file_path)


class IImageLoader(ABC):
    def __init__(self, path: str, files_finder: IFindFiles = FindFiles(), loader: ILoader = OpenCvLoader()):
        self.path = path
        self.files_finder = files_finder
        self.images = []
        self.loader = loader

    def load(self):
        ...


class ImageLoader(IImageLoader):
    def load(self):
        self.images = [self.loader.load(file) for file in self.files_finder.find(self.path)]
