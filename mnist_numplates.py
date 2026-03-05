#!/usr/bin/env python3

'''
From https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/
'''

import os
import gzip

import requests
import numpy as np
import matplotlib.pyplot as plt

# Source URLs and data dir
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}
data_dir = "mnist"

# Download dataset if not present
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    for fname in data_sources.values():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print("Downloading file: " + fname)
            resp = requests.get(base_url + fname, stream=True)
            resp.raise_for_status()  # Ensure download was succesful
            with open(fpath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=128):
                    fh.write(chunk)

# Load dataset
mnist_dataset = {}
# Images
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)
# Labels
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)

# Make a N-digit image
indices = [0, 1, 2, 3] # We will take the images from these indices to form a multi-digit image
final_image = np.zeros(shape=(28 * len(indices), 28), dtype=np.uint8).flatten() # Flat and transposed pixel array so we can easily copy flattened pixels in order
for i, index in enumerate(indices):
    mnist_image = mnist_dataset["training_images"][index, :].reshape(28, 28).transpose().flatten()  # Flat and transposed
    final_image[784*i:784*(i+1)] = mnist_image[:] # Copy digit pixels: First digit is 0:784, second is 784:1568, etc.
final_image = final_image.reshape(28 * len(indices), 28).transpose() # Reshape and transpose to get the digits horizontally

# Get digit labels just to show
plate_number = ''
for index in indices:
    plate_number += str(mnist_dataset["training_labels"][index])

# Show final image
plt.title(plate_number)
plt.imshow(final_image, cmap="gray")
plt.show()
