import numpy as np
from keras import Sequential
from keras.src.saving import load_model

from app.images import ImageLoader, load_labels_and_filenames
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

    def train(self, annotations_path: str, path: str, mapping: dict[str, int]):
        labels, filenames = load_labels_and_filenames(annotations_path, mapping)

        image_loader = ImageLoader(path)
        images, filenames = image_loader.load()

        open_cv_preprocessing = OpenCvPreprocessing(images)
        images = open_cv_preprocessing.preprocess()

        splitter = Splitter(images, labels)
        splitter.split(0.8)
        splitter.to_categorical(len(mapping.keys()))

        train_x, train_y, test_x, test_y = splitter.train_x, splitter.train_y, splitter.test_x, splitter.test_y

        training = Training(train_x, train_y, test_x, test_y)
        model = training.train(epochs=20, batch_size=32)

        self.model = model

    @staticmethod
    def _preprocess_image(image: np.array) -> np.array:
        preprocessing = OpenCvPreprocessing([image])
        image = preprocessing.preprocess()[0]
        return image

    def predict(self, image: np.array) -> tuple[list[float], str]:
        image_preprocessed = self._preprocess_image(image)
        predictions = self.model.predict(image_preprocessed.reshape(1, 300, 300, 3))[0]

        return predictions, self.labels[np.argmax(predictions)]


model = Model(["degrade", "degradant", "aucun-rapport"])
model.load("degradai.keras")

def get_model() -> Model:
    return model

def get_labels() -> list[str]:
    return model.labels