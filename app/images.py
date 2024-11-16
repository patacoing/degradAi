import json
from abc import ABC, abstractmethod
import cv2
import numpy as np
import os


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
    def load(self) -> list[tuple[np.array, str]]:
        filenames = os.listdir(self.path)
        images = []

        for filename in filenames:
            if filename == ".DS_Store":
                continue

            if os.path.isfile(f"{self.path}/{filename}"):
                images.append(self.loader.load(f"{self.path}/{filename}"))

        return images, filenames


def load_labels_and_filenames(annotation_path: str, mapping: dict[str, int]) -> tuple[list[str], list[str]]:
    with open(annotation_path, "r") as file:
        data_json = json.load(file)
        labels = []
        filenames = []

        for annotation in data_json["annotations"]:
            filename = annotation["fileName"]
            label = annotation["annotation"]["label"]

            if filename in filenames:
                continue

            labels.append(mapping[label])
            filenames.append(filename)

    labels = np.array(labels)
    return labels, filenames
