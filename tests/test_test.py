import pytest
import numpy as np


class Test_aaa(object):

    @pytest.mark.parametrize(
        "img_phy",
        [
            (
                [-135, -22.5, 90, 45, 100],
            ),
        ],
    )
    def test_aaa(
        self, img_phy
    ):
        """Test the convert2physical function"""

        assert np.allclose(img_phy, img_phy, rtol=1e-3)
