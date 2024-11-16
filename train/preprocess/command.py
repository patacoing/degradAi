import argparse
from pathlib import Path
from PIL import Image
import mlflow

from app.images import ImageLoader
from app.preprocessing import OpenCvPreprocessing
from .directory_hash import hash_dir

parser = argparse.ArgumentParser("preprocess")
parser.add_argument("--images_input", type=str)
parser.add_argument("--images_output", type=str)
parser.add_argument("--hash_output", type=str)

args = parser.parse_args()
images_input = args.images_input
images_output = args.images_output
hash_output = args.hash_output

print("loading images")
image_loader = ImageLoader(images_input)
images, filenames = image_loader.load()
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


computed_hash = hash_dir(images_output)
with open(str(Path(hash_output) / "hash.txt"), "w") as file:
    file.write(computed_hash)

console_output = f""" 
    number_images_output: {len(filenames)}
    computed_hash: {computed_hash}"""

# mlflow.log_metric("number_files_input", 1)
# mlflow.log_metric("number_images_output", 2)