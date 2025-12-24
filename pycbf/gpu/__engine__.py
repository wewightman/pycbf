import cupy as cp
import numpy as np
from pycbf.__bf_base_classes__ import BeamformerException
import os

if not cp.cuda.is_available(): raise BeamformerException("Unable to find a CUDA compatible device")

__eng_dir__ = os.path.join(os.path.dirname(__file__), "__engine__.cu")
with open(__eng_dir__, mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)

xInfo = np.dtype([('x0', np.float32),('dx', np.float32),('nx', np.int32)])
RFInfo = np.dtype([('ntx', np.int32),('nrx', np.int32),('ndim', np.int32),('tInfo', xInfo)])

das_bmode_cubic_synthetic = module.get_function("das_bmode_cubic_synthetic")
das_bmode_rxseparate_cubic_synthetic = module.get_function("das_bmode_rxseparate_cubic_synthetic")
das_bmode_cubic_tabbed = module.get_function("das_bmode_cubic_tabbed")
das_bmode_rxseparate_cubic_tabbed = module.get_function("das_bmode_rxseparate_cubic_tabbed")