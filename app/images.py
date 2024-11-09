import json
from abc import ABC, abstractmethod
import cv2
import numpy as np


class ILoader(ABC):
    @classmethod
    @abstractmethod
    def load(cls, file_path: str):
        raise NotImplementedError


class OpenCvLoader:
    @classmethod
    def load(cls, file_path: str) -> np.array:
        return cv2.imread(file_path)


class IImageLoader(ABC):
    def __init__(self, path: str, loader: ILoader = OpenCvLoader()):
        self.path = path
        self.images = []
        self.loader = loader

    @abstractmethod
    def load(self, filenames: list[str]) -> list[np.array]:
        raise NotImplementedError


class ImageLoader(IImageLoader):
    def load(self, filenames: list[str]) -> list[np.array]:
        return [self.loader.load(f"{self.path}/{filename}") for filename in filenames]


def load_labels_and_filenames(annotation_path: str, mapping: dict[str, int]) -> tuple[list[str], list[str]]:
    with open(annotation_path, "r") as file:
        data_json = json.load(file)
        labels = [annotation["annotation"]["label"] for annotation in data_json["annotations"]]
        filenames = [label["fileName"] for label in data_json["annotations"]]

    labels = np.array([mapping[label] for label in labels])
    return labels, filenames
