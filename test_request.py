import io
import os
import sys

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

IMAGENET_PATH = '/Users/felixsu/Home/school/Sp18/CS194/project/data/nips'
IMG_ID = sys.argv[1]
# The name of the image file to annotate
file_name = os.path.join(os.path.join(IMAGENET_PATH, 'dev'), str(IMG_ID) + ".png")

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image, max_results=20)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)
