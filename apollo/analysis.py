import numpy as np
import matplotlib.pyplot as plt
from apollo.config import Params

params = Params()

__all__ = ["convert2physical", "size_frequency_compare_plot"]


def convert2physical(labels, subimg_pix, indices):
    """
    Return a set of physical parameter of craters, including size and location.

    Parameters
    ----------
    img_phy: array or array-like
        A set of parameters contain real-world, physical image center
        longitude, latitude, image width, height in degrees, and
        resolution, all given by the user
    img_pix: array or array-like
        The number of pixels in each picture.
    labels: array or array-like
        The predicted location and size of craters in the picture.
    subimg_pix: array or array-like
        The number of pixels in each sub-picture.
    indices: array or array-like
        The indices of different sub-pictures.

    Returns
    -------
    crater_size: array or array-like
        The physical, real-world crater sizes
    crater_loc: array or array-like
        The physical, real-world crater locations

    Examples
    --------
    >>> import apollo
    >>> labels = [0.782866827, 0.463593897, 0.099159444, 0.099098558]
    >>> apollo.convert2physical([-135, -22.5, 90, 45, 100],
    [27291, 13645], labels, [416, 416], [0, 1])
    """

    x, y, w, h = labels
    subimg_pix_w, subimg_pix_h = subimg_pix
    part, a, b = indices
    phy_lon = params.MOON_LOC[part][0]
    phy_lat = params.MOON_LOC[part][1]

    # longitude of the pic origin
    img_origin_x = phy_lon - 0.5 * params.MOON_TRAIN_W
    img_origin_y = phy_lat + 0.5 * params.MOON_TRAIN_H

    # longitude of the subpic origin
    subimg_origin_x = img_origin_x + a * subimg_pix_w \
        * params.MOON_RESO * 180 / (np.pi * params.MOON_RADIUS)
    subimg_origin_y = img_origin_y - b * subimg_pix_h \
        * params.MOON_RESO * 180 / (np.pi * params.MOON_RADIUS)

    # longitude of the crater center
    crater_lon = subimg_origin_x + x * subimg_pix_w \
        * params.MOON_RESO * 180 / (np.pi * params.MOON_RADIUS)
    crater_lat = subimg_origin_y - y * subimg_pix_h \
        * params.MOON_RESO * 180 / (np.pi * params.MOON_RADIUS)

    crater_h = h * subimg_pix_h * params.MOON_RESO / 1e3

    crater_size = crater_h

    return crater_lon, crater_lat, crater_size


def size_frequency_compare_plot(folderpath, detection, real):
    """
    Plot a separate plot of the cumulative crater size-frequency
    distribution of detected craters, if information to calculate
    crater size is provided.

    Parameters
    ----------
    folderpath: string
        The user-specified input folder location
    detection: array or array-like
        The physical, real-world crater sizes
    real: array or array-like
        The physical, real-world crater sizes
    """
    countdetected = np.histogram(detection, np.arange(1, 100, 2))
    xdetected = list(countdetected[1])
    ydetected = list(countdetected[0])
    countreal = np.histogram(real, np.arange(1, 100, 2))
    xreal = list(countreal[1])
    yreal = list(countreal[0])
    yreal.append(0)
    ydetected.append(0)
    residual = [a_item - b_item for a_item, b_item in zip(ydetected, yreal)]

    plt.subplot(2, 1, 1)

    plt.plot(xdetected, ydetected, "-co", label="real_crater_size1")
    plt.plot(xreal, yreal, "-kx", label="crater_detector_algrithom")
    plt.xlabel("$Diameter, km$", fontsize=14)  # Add an x-label to the axes.
    plt.ylabel("$Number of craters$", fontsize=14)
    plt.title(
        "Crater Size-frequency Distribution", fontsize=14
    )  # Add a title to the axes.
    plt.legend()  # Add a legend.

    plt.subplot(2, 1, 2)
    plt.plot(xdetected, residual, "-ro", label="real_crater_size1")
    plt.ylabel("$Residuals$", fontsize=14)

    filepath = folderpath + "/size_frequency.png"
    plt.savefig(filepath)

    return 1
