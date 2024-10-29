from app.images import ImageLoader
from app.preprocessing import OpenCvPreprocessing
from app.splitter import Splitter

PATH = "./data"

image_loader = ImageLoader(PATH)
image_loader.load()

open_cv_preprocessing = OpenCvPreprocessing(image_loader.images)
open_cv_preprocessing.preprocess()

images = open_cv_preprocessing.images

splitter = Splitter(images)
splitter.split(0.8)
train_images, test_images = splitter.train, splitter.test