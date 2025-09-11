"""Python wrapper for c-based 1D cubic interpolators"""
import ctypes as ct
import numpy as np
from glob import glob
import platform as _pltfm
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# determine installed path
dirpath = os.path.dirname(__file__)

matptr = ct.POINTER(ct.POINTER(ct.c_float))

# determine the OS and relative binary file
if _pltfm.uname()[0] == "Windows":
    res = glob(os.path.abspath(os.path.join(dirpath, "*.dll")))
    name = res[0]
elif _pltfm.uname()[0] == "Linux":
    res = glob(os.path.abspath(os.path.join(dirpath, "*.so")))
    name = res[0]
else:
    res = glob(os.path.abspath(os.path.join(dirpath, "*.dylib")))
    name = res[0]

# load the c library
__cpu_pycbf__ = ct.CDLL(name)

__cpu_pycbf__.beamform.argtypes = (
    ct.c_float, 
    ct.c_float, 
    ct.c_int, 
    ct.POINTER(ct.c_float), 
    ct.c_int, 
    ct.c_float,
    ct.POINTER(ct.c_float), 
    ct.POINTER(ct.c_float), 
    ct.POINTER(ct.c_float), 
    ct.POINTER(ct.c_float), 
    ct.POINTER(ct.c_float)
)
__cpu_pycbf__.beamform.restype = (None)
__cpu_pycbf__.beamform.__doc__ = """beamform an aline's worth of data"""