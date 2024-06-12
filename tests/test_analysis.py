import pytest
import numpy as np
from apollo.analysis import convert2physical, size_frequency_compare_plot


class Test_analysis(object):
    """
    Class for testing the functions within analysis.py.
    """

    @pytest.mark.parametrize(
        "labels, subimg_pix, \
            indices, crater_size, crater_lon, crater_lat",
        [
            (
                [0.782866827, 0.463593897, 0.099159444, 0.099098558],
                [416, 416],
                ['A', 0, 1],
                4.1225,
                -178.926,
                -2.00795,
            ),
        ],
    )
    def test_convert2physical(
        self, labels, subimg_pix,
        indices, crater_size, crater_lon, crater_lat
    ):
        """Test the convert2physical function"""

        my_crater_lon, my_crater_lat, my_crater_size = convert2physical(
            labels, subimg_pix, indices
        )

        assert np.allclose(my_crater_lon, crater_lon, rtol=1e-3)

        assert np.allclose(my_crater_lat, crater_lat, rtol=1e-3)

        assert np.allclose(my_crater_size, crater_size, rtol=1e-3)

    @pytest.mark.parametrize(
        "folderpath, detection, real",
        [
            (
                "..",
                [3.0, 4.0, 4.0, 5.0, 5.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [3.0, 4.0, 4.0, 5.5, 7.0, 8.0, 9.0, 10.0, 11.0, 11.0],
            ),
        ],
    )
    def test_size_frequency_compare_plot(self, folderpath, detection, real):
        """Test the size_frequency_compare_plot function"""
        # assert 1
        assert size_frequency_compare_plot(folderpath, detection, real)
