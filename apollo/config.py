class Params:
    """
    Class to define some necessary global variables in our project.

    Parameters
    ----------
    MOON_RADIUS: float
        The radius of the Moon(units: m).
    MOON_LOC: dict
        The dictionary about the latitude and longitude
    of the central point of the original four
    picture of the Moon(units: degree).
    MOON_TRAIN_W: int
        The longitude that this image spans(units: degree).
    MOON_TRAIN_H: int
        The latitude that this image spans(units: degree).
    MOON_RESO: int
        The resolution of the picture, which means
    how many meters a pixel can represent(units: meters).
    LOG_LEVEL: str
        The log level.
    COLOR:
        The color we might use in our project.

    """

    MOON_RADIUS: float = 1.7374e6  # unit: m

    MOON_LOC: dict = {
        "A": [-135, -22.5],
        "B": [-135, 22.5],
        "C": [-45, -22.5],
        "D": [-45, 22.5],
    }  # units: degree

    MOON_TRAIN_W: int = 90
    MOON_TRAIN_H: int = 45  # units: degree

    MOON_RESO: int = 100  # units: m/px

    LOG_LEVEL: str = "INFO"

    COLOR = [
        "red",
        "orange",
        "yellow",
        "blue",
        "cyan",
        "azure",
        "lime",
        "lavender",
        "ghostwhite",
    ]
