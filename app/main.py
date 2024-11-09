import pathlib

from app.images import ImageLoader, load_labels_and_filenames
from app.model import Model
from app.preprocessing import OpenCvPreprocessing
from app.splitter import Splitter
from app.training import Training


PATH = str(pathlib.Path(__file__).parent.parent.resolve()) + "/data"
ANNOTATION_PATH = str(pathlib.Path(__file__).parent.parent.resolve()) + "/degrade-Degradant-annotations.json"

mapping = {
        "degrade": 0,
        "degradant": 1,
        "aucun-rapport": 2
    }

if False:
    model = Model(list(mapping.keys()))
    model.load("degradai.keras")

else:
    model = Model(list(mapping.keys()))

    model.train(ANNOTATION_PATH, PATH, mapping)
    model.save("degradai.keras")


predictions, label = model.predict("wow", "marius.jpg")

print(predictions, label)