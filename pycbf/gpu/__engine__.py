import cupy as cp
import numpy as np
import os

__eng_dir__ = os.path.join(os.path.dirname(__file__), "__engine__.cu")
with open(__eng_dir__, mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)

xInfo = np.dtype([('x0', np.float32),('dx', np.float32),('nx', np.int32)])
RFInfo = np.dtype([('ntx', np.int32),('nrx', np.int32),('ndim', np.int32),('tInfo', xInfo)])

das_bmode_cubic = module.get_function("das_bmode_cubic")
das_bmode_rxseparate_cubic = module.get_function("das_bmode_rxseparate_cubic")