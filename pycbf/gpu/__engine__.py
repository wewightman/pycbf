import cupy as cp
import os

__eng_dir__ = os.path.join(os.path.dirname(__file__), "__engine__.cu")
with open(__eng_dir__, mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)

das_bmode_cubic = module.get_function("das_bmode_cubic")