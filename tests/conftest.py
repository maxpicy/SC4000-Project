import pytest
import pathlib

DATA_ROOT = pathlib.Path("kaggle_dataset")
SAMPLE_TRIP = DATA_ROOT / "train" / "2020-05-15-US-MTV-1"


@pytest.fixture
def sample_device_dir():
    # Return first device folder found under the sample trip.
    dirs = [d for d in SAMPLE_TRIP.iterdir() if d.is_dir()]
    assert dirs, "No device dirs found under sample trip"
    return dirs[0]
