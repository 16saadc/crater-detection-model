from moon_picture_slicing import crop_func
import unittest
import os
import numpy as np
import cv2


class TestCropFunc(unittest.TestCase):
    def test_crop_func(self):

        img = cv2.imread("Moon_WAC_Training/images/Lunar_A.jpg")
        h, w = img.shape[:2]
        crop_area = 'A'
        crop_size = 416
        label =\
            'Moon_WAC_Training/labels/lunar_crater_database_robbins_train.csv'
        crop_func(crop_area=crop_area,
                  crop_size=crop_size,
                  image_location='Moon_WAC_Training/images/Lunar_A.jpg',
                  label_location=label,
                  filepath='Moon_WAC_Training')

        # Check if the directories are created
        self.assertTrue(os.path.exists('Moon_WAC_Training/sliced_images'))
        self.assertTrue(os.path.exists('Moon_WAC_Training/sliced_labels'))

        # Check if images are saved correctly
        for i in range(np.int64(np.floor(w/crop_size))+1):
            for j in range(np.int64(np.floor(h/crop_size))+1):
                image_path =\
                    'Moon_WAC_Training/sliced_images/{}-{}-{}.jpg'.format(
                        crop_area, i, j)
                self.assertTrue(os.path.exists(image_path))

        # Check if csv files are saved correctly
        for i in range(np.int64(np.floor(w/crop_size))+1):
            for j in range(np.int64(np.floor(h/crop_size))+1):
                csv_path =\
                    'Moon_WAC_Training/sliced_labels/{}-{}-{}.csv'.format(
                        crop_area, i, j)
                self.assertTrue(os.path.exists(csv_path))


if __name__ == '__main__':
    unittest.main()
