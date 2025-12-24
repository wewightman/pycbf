from typing import Any
import h5py as h5

import logging
logger = logging.getLogger(__name__)

# HDF5 files
def __help_loadHDF5asdict__(data:h5.Dataset|h5.Group|h5.File) -> dict | Any:
    """Load the passed HDF5 dataset as a dictionary"""
    if isinstance(data, h5.Dataset):
        try:
            return data.asstr()[()]
        except TypeError as te:
            return data[()]
    else:
        group = {}
        for name, node in data.items():
            group[name] = __help_loadHDF5asdict__(node)
        return group

def loadHDF5asdict(file):
    """Load the given hdf5 file as a dictionary"""
    with h5.File(file, mode='r') as fp:
        return __help_loadHDF5asdict__(fp)
    
def __help_putdictasHDF5__(src:dict|Any, name:str, dst:h5.Group|h5.File):
    """Recursively put all items from src dictionary into dst hdf5"""
    if isinstance(src, dict):
        grp = dst.create_group(name, track_order=True)
        for subname, subsrc in src.items():
            __help_putdictasHDF5__(subsrc, subname, grp)
    else:
        try:
            dst[name] = src
        except TypeError as e:
            dst[name] = str(src)

def putdictasHDF5(file, data:dict):
    with h5.File(file, mode='w') as fp:
        for key, value in data.items():
            __help_putdictasHDF5__(value, key, fp)