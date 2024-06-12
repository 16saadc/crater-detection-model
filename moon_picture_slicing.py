import pandas as pd
import numpy as np
import csv
import cv2
import math
import os


def crop_func(crop_area, crop_size, image_location, label_location, filepath):
    """
    Slice/crop an image and crater label, to sub-images of specific pixel size
    and csv file of crater label for each sub-image, and save them into
    a folder specified by user.

    Parameters
    ----------
    crop_area: str
        image of moon for slicing. Options : 'A', 'B', 'C', or 'D'.
    crop_size: int
        The number of pixels for sub-images.
    image_location: str
        Filepath of input image (image file).
    label_location: str
        Filepath of input labels (csv file).
    filepath: str
        Filepath to save the sub-images and labels created by the function.
        On that particular filepath the user passed, 2 new folders
        named "sliced_images" and "sliced_labels" are created automatically.

    Example
    ---------
    crop_func(crop_area='A',
              crop_size=416,
              image_location='Moon_WAC_Training/images/Lunar_A.jpg',
              label_location='Moon_WAC_Training/labels/lunar_crater_database_robbins_train.csv',
              filepath='Moon_WAC_Training')
    """

    # create a new folders
    os.mkdir(filepath + "/sliced_images")
    os.mkdir(filepath + "/sliced_labels")

    # Load the image and csv file
    img = cv2.imread(image_location)
    df = pd.read_csv(label_location)

    # Divide df into ABCD file
    if crop_area == "A":
        df = df[
            (df["LAT_CIRC_IMG"] < 0) & (df["LON_CIRC_IMG"] < -90)
        ].reset_index()
        init_lon, init_lat = -180, 0
    elif crop_area == "B":
        df = df[
            (df["LAT_CIRC_IMG"] > 0) & (df["LON_CIRC_IMG"] < -90)
        ].reset_index()
        init_lon, init_lat = -180, 45
    elif crop_area == "C":
        df = df[
            (df["LAT_CIRC_IMG"] < 0) & (df["LON_CIRC_IMG"] > -90)
        ].reset_index()
        init_lon, init_lat = -90, 0
    elif crop_area == "D":
        df = df[
            (df["LAT_CIRC_IMG"] > 0) & (df["LON_CIRC_IMG"] > -90)
        ].reset_index()
        init_lon, init_lat = -90, 45
    else:
        return "Wrong crop area"

    # Calculate the size of each sub-image
    h, w, _ = img.shape
    sub_h, _ = crop_size, crop_size

    x = (crop_size * 90) / w
    y = (crop_size * 45) / h

    # Cut the image into sub-images
    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):
            sub_img = img[i: i + crop_size, j: j + crop_size]
            if sub_img.shape[0] != crop_size or sub_img.shape[1] != crop_size:
                # Fill the empty part with black color
                sub_img = cv2.copyMakeBorder(
                    sub_img,
                    0,
                    sub_h - sub_img.shape[0],
                    0,
                    crop_size - sub_img.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0],
                )

            cv2.imwrite(
                filepath
                + "/sliced_images/{}-{}-{}.jpg".format(
                    crop_area, int(j / crop_size), int(i / crop_size)
                ),
                sub_img,
            )

    # Cut the csv file data into corresponding sub_image
    for i in range(np.int64(np.floor(w / crop_size)) + 1):
        for j in range(np.int64(np.floor(h / crop_size)) + 1):
            dftemp = df[
                (df["LON_CIRC_IMG"] > (i * x) + init_lon)
                & (df["LON_CIRC_IMG"] <= ((i + 1) * x) + init_lon)
                & (df["LAT_CIRC_IMG"] < (-j * y + init_lat))
                & (df["LAT_CIRC_IMG"] >= (-(j + 1) * y + init_lat))
            ].reset_index()

            with open(
                filepath
                + "/sliced_labels/{}-{}-{}.csv".format(
                    crop_area, int(i), int(j)),
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow([])

            for n in range(len(dftemp)):
                x_data = (dftemp["LON_CIRC_IMG"][n] - ((i * x) + init_lon)) / x
                y_data = -(dftemp["LAT_CIRC_IMG"][n] - (-j * y + init_lat)) / y
                h_data = dftemp["DIAM_CIRC_IMG"][n] / (crop_size / 10)
                w_data = h_data / np.cos(
                    math.radians(abs(dftemp["LAT_CIRC_IMG"][n])))

                lat_data = dftemp["LAT_CIRC_IMG"][n]
                lon_data = dftemp["LON_CIRC_IMG"][n]
                diam_data = dftemp["DIAM_CIRC_IMG"][n]

                data = [
                    x_data, y_data, w_data, h_data, lat_data,
                    lon_data, diam_data]

                if w_data < 1:
                    with open(
                        filepath
                        + "/sliced_labels/{}-{}-{}.csv".format(
                            crop_area, int(i), int(j)
                        ),
                        "a",
                        newline="",
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

    return None
