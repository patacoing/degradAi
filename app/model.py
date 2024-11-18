import numpy as np
from keras import Sequential
from keras.src.saving import load_model

from app.images import ImageLoader, load_labels_and_filenames, load_labels_and_filesnames_from_txt
from app.preprocessing import OpenCvPreprocessing
from app.splitter import Splitter
from app.training import Training


class Model:
    def __init__(self, labels: list[str]):
        self.model = Sequential()
        self.labels = labels


    def load(self, path: str):
        self.model = load_model(path)

    def save(self, path: str):
        self.model.save(path)

    def train(self, train_annotations_path: str, train_path: str, test_annotations_path: str, test_path: str, mapping: dict[str, int]):
        train_labels = load_labels_and_filesnames_from_txt(train_annotations_path)
        test_labels = load_labels_and_filesnames_from_txt(test_annotations_path)

        image_loader = ImageLoader(train_path)
        train_images, filenames = image_loader.load()

        image_loader = ImageLoader(test_path)
        test_images, filenames = image_loader.load()

        open_cv_preprocessing = OpenCvPreprocessing(train_images)
        train_images = open_cv_preprocessing.preprocess()

        open_cv_preprocessing = OpenCvPreprocessing(test_images)
        test_images = open_cv_preprocessing.preprocess()

        # splitter = Splitter(images, labels)
        # splitter.split(0.8)
        # splitter.to_categorical(len(mapping.keys()))

        train_x, train_y, test_x, test_y = train_images, train_labels, test_images, test_labels

        training = Training(train_x, train_y, test_x, test_y)
        model = training.train(epochs=1, batch_size=32)

        self.model = model

    @staticmethod
    def _preprocess_image(image: np.array) -> np.array:
        preprocessing = OpenCvPreprocessing([image])
        image = preprocessing.preprocess()[0]
        return image

    def predict(self, image: np.array, verbose: int = 0) -> tuple[list[float], str]:
        image_preprocessed = self._preprocess_image(image)
        predictions = self.model.predict(image_preprocessed.reshape(1, 300, 300, 3), verbose=verbose)[0]

        return predictions, self.labels[np.argmax(predictions)]


model = Model(["degrade", "degradant", "aucun-rapport"])
model.load("degradai.keras")

def get_model() -> Model:
    return model

def get_labels() -> list[str]:
    return model.labels