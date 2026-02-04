import pytest

@pytest.fixture(scope='session')
def interpolator_groundtruth_data():
    """Loads raw traces and expected interpolated data into a dictionary"""
    from pycbf.dataio.__hdf5_engines__ import loadHDF5asdict

    return loadHDF5asdict("./data/interpolator_groundtruth_data.h5")