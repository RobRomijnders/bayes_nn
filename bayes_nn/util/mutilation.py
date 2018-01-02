"""
This file contains all the mutilation functions that will be used for the experiments.

To use the mutilation function, define it in `development.config.ini` in the [DEFAULT] section
"""

from PIL import Image
import numpy as np
from bayes_nn import conf


def rotation(images, angle):
    """
    Rotates the image over <angle> degrees.
    :param images: numpy array
    :param angle:
    :return: numpy array
    """
    num_batch = images.shape[0]
    for n in range(num_batch):
        im = Image.fromarray(np.reshape(images[n], (28,28)))
        im = im.rotate(-1*angle)
        im = np.expand_dims(np.array(im), axis=0)
        images[n] = im
    return images


def noise(images, sigma):
    images += sigma * np.random.randn(*images.shape)  # , *conf.range)
    return images


def noise_clip(images, sigma):
    images += np.clip(sigma * np.random.randn(*images.shape), *conf.range)
    return images
