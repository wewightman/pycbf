"""Python wrapper for c-based 1D cubic interpolators"""
import ctypes as ct
import numpy as np
from glob import glob
import platform as _pltfm
import os
from pycbf.__bf_base_classes__ import __PYCBF_DATATYPE__

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# determine installed path
dirpath = os.path.dirname(__file__)

if type(__PYCBF_DATATYPE__) == type(np.float32):
    __pycbf_ctype__ = ct.c_float
    __pycbf_pnt__   = ct.POINTER(__pycbf_ctype__)
elif type(__PYCBF_DATATYPE__) == type(np.float64):
    __pycbf_ctype__ = ct.c_double
    __pycbf_pnt__   = ct.POINTER(__pycbf_ctype__)
else:
    raise Exception("Unknown dtype used for pycbf datatype")

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

__cpu_pycbf__.beamform_cubic.argtypes = (
    __pycbf_ctype__, 
    __pycbf_ctype__, 
    ct.c_int, 
    __pycbf_pnt__, 
    ct.c_int, 
    __pycbf_ctype__,
    __pycbf_pnt__, 
    __pycbf_pnt__, 
    __pycbf_pnt__, 
    __pycbf_pnt__, 
    __pycbf_pnt__
)
__cpu_pycbf__.beamform_cubic.restype = (None)
__cpu_pycbf__.beamform_cubic.__doc__ = """beamform an aline's worth of data using cubic interpolation for each pixel"""

__cpu_pycbf__.beamform_nearest.argtypes = (
    __pycbf_ctype__, 
    __pycbf_ctype__, 
    ct.c_int, 
    __pycbf_pnt__, 
    ct.c_int, 
    __pycbf_ctype__,
    __pycbf_pnt__, 
    __pycbf_pnt__, 
    __pycbf_pnt__, 
    __pycbf_pnt__, 
    __pycbf_pnt__,
    ct.c_int
)
__cpu_pycbf__.beamform_nearest.restype = (None)
__cpu_pycbf__.beamform_nearest.__doc__ = """beamform an aline's worth of data using nearest neighbor interpolation for each pixel with optional upsampling"""
