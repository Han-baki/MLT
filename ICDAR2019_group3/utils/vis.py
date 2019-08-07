"""Tools for vizualisation of convolutional neural network filter in keras models."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools




def to_rec(box, image_size):
    """Finds minimum rectangle around some points and scales it to desired 
    image size.
    
    # Arguments
        box: Box or points [x1, y1, x2, y2, ...] with values between 0 and 1.
        image_size: Size of output image.
    # Return
        xy_rec: Corner coordinates of rectangle, array of shape (4, 2).
    """
    img_height, img_width = image_size
    xmin = np.min(box[0::2]) * img_width
    xmax = np.max(box[0::2]) * img_width
    ymin = np.min(box[1::2]) * img_height
    ymax = np.max(box[1::2]) * img_height
    xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return xy_rec



