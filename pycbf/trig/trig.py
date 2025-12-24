import platform as _pltfm
import ctypes
from glob import glob
from pathlib import Path
import os

dirpath = os.path.dirname(__file__)

# determine the OS
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
_trig = ctypes.CDLL(name)

# c function definitions inputs and outputs
_trig.pwtxengine.argtypes = (
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
)
pwtxengine = _trig.pwtxengine

_trig.rxengine.argtypes = (
    ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
)
rxengine = _trig.rxengine

_trig.genmask3D.argtypes = (
    ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_int, 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
)
genmask3D = _trig.genmask3D

_trig.calcindices.argtypes = (
    ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
)
calcindices = _trig.calcindices

_trig.selectdata.argtypes = (
    ctypes.c_int, ctypes.POINTER(ctypes.c_int), 
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
)
selectdata = _trig.selectdata

# copysubvec(int Norig, int Nsub, int index, float *orig, float *sub)
_trig.copysubvec.argtypes = (
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
)
copysubvec = _trig.copysubvec

_trig.fillarr.argtypes = (
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_float
)
fillarr = _trig.fillarr

_trig.sumvecs.argtypes = (
    ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.POINTER(ctypes.c_float)
)
sumvecs = _trig.sumvecs

_trig.printifa.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float))
printifa = _trig.printifa
