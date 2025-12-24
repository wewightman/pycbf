import cupy as cp
import numpy as np
from pycbf.__bf_base_classes__ import BeamformerException
import os

if not cp.cuda.is_available(): raise BeamformerException("Unable to find a CUDA compatible device")

# load the basic beamformer module
__base_eng_path__ = os.path.join(os.path.dirname(__file__), "__engines__.cu")
with open(__base_eng_path__, mode='r') as fp: raw_module = fp.read()
__base_eng__ = cp.RawModule(code=raw_module)

xInfo = np.dtype([('x0', np.float32),('dx', np.float32),('nx', np.int32)])
RFInfo = np.dtype([('ntx', np.int64),('nrx', np.int64),('ndim', np.int32),('tInfo', xInfo)])

das_bmode_synthetic_korder_cubic = __base_eng__.get_function("das_bmode_synthetic_korder_cubic")
das_bmode_synthetic_multi_interp = __base_eng__.get_function("das_bmode_synthetic_multi_interp")
das_bmode_tabbed_korder_cubic    = __base_eng__.get_function("das_bmode_tabbed_korder_cubic")
das_bmode_tabbed_multi_interp    = __base_eng__.get_function("das_bmode_tabbed_multi_interp")

# load the advanced beamfroming code
__advacned_eng_path__ = os.path.join(os.path.dirname(__file__), "__advanced_engines__.cu")
with open(__advacned_eng_path__, mode='r') as fp: raw_module = fp.read()
__advacned_eng__ = cp.RawModule(code=raw_module)

calc_dmas = __advacned_eng__.get_function("calc_dmas")
