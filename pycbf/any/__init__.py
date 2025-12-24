"""Wrapper module to automatically import GPU beamformers if a CUDA ready GPU is available, or CPU beamformers if not"""

try:
    import cupy as cp
    if cp.is_available():
        from pycbf.gpu import *
    else: from pycbf.cpu import *
except: from pycbf.cpu import *