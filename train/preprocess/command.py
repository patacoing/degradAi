import argparse
from pathlib import Path
from PIL import Image
import mlflow

from app.images import ImageLoader, isfile, listfiles
from app.preprocessing import OpenCvPreprocessing

parser = argparse.ArgumentParser("preprocess")
parser.add_argument("--images_input", type=str)
parser.add_argument("--images_output", type=str)

args = parser.parse_args()
images_input = args.images_input
images_output = args.images_output

print("loading images")
image_loader = ImageLoader(images_input)
images, filenames = image_loader.load(is_file=isfile, list_files=listfiles)
print("images loaded")

print("preprocessing images")
open_cv_preprocessing = OpenCvPreprocessing(images)
images = open_cv_preprocessing.preprocess()
print("images preprocessed")

print("saving images")
images = (images * 255).astype("uint8")
for image, filename in zip(images, filenames):
    im = Image.fromarray(image)
    im.save(f"{images_output}/{filename}")
print("images saved")


console_output = f""" 
    number_images_output: {len(filenames)}"""

# mlflow.log_metric("number_files_input", 1)
# mlflow.log_metric("number_images_output", 2)