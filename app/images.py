import json
from abc import ABC, abstractmethod
from typing import Callable
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


def isfile(file_path: str) -> bool:
    return os.path.isfile(file_path)

def listfiles(path: str) -> list[str]:
    return os.listdir(path)


class ImageLoader(IImageLoader):
    def load(
        self,
        is_file: Callable[[str], bool],
        list_files: Callable[[str], list[str]]
    ) -> list[tuple[np.array, str]]:
        filenames = list_files(self.path)
        images = []

        for filename in filenames:
            if filename == ".DS_Store":
                continue

            if is_file(f"{self.path}/{filename}"):
                images.append(self.loader.load(f"{self.path}/{filename}"))

        return images, filenames

def load_labels_and_filesnames_from_txt(path: str) -> tuple[list[str], list[str]]:
    with open(path, "r") as file:
        data = file.readlines()
        labels = []

        for line in data:
            line = line.strip()
            label = line
            labels.append(label)

    return labels

def load_labels_and_filenames(
    annotation_path: str,
    mapping: dict[str, int]
) -> tuple[list[str], list[str]]:

    with open(annotation_path, "r") as file:
        data_json = json.load(file)
        labels = []
        filenames = []

        for annotation in data_json.get("annotations", []):
            filename = annotation.get("fileName", "")
            label = annotation.get("annotation", {}).get("label", "")

            if not label or not filename:
                raise ValueError("Invalid annotation")

            if filename in filenames:
                continue

            labels.append(mapping[label])
            filenames.append(filename)

    labels = np.array(labels)
    return labels, filenames