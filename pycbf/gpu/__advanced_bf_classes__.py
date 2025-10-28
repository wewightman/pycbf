from pycbf.__bf_base_classes__ import __BMFRM_PARAMS__
from pycbf.__advanced_bf_classes__ import DMASBeamformer
from pycbf.gpu.__core_bf_classes__ import TabbedBeamformer, SyntheticBeamformer, __GPU_Beamformer__
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from numpy import ndarray as npNDArray
from cupy  import ndarray as cpNDArray

@dataclass(kw_only=True)
class __GPU_DMAS_Beamformer__(DMASBeamformer):

    def __post_init__(self):
        DMASBeamformer.__post_init__(self)

    def __run_base_beamforming__(self, txrxt):
        raise NotImplementedError("Instances of __GPU_DMAS_Beamfomer__ must implement __run_base_beamforming__ method")

    def __check_or_init_buffer__(self, buffer : cpNDArray | None = None) -> cpNDArray:
        import cupy as cp

        if self.sumlags: shape = self.nop
        else:            shape = (self.nlags, self.nop)

        if buffer is None: imout = cp.zeros(shape, dtype=cp.float32)
        else: raise Exception("Something is wrong with input buffers")

        return imout
    
    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:
        from pycbf.gpu.__engines__ import calc_dmas
        import numpy as np
        import cupy as cp

        # delay and sum the data along either the tx or rx axes
        imbf = self.__run_base_beamforming__(txrxt)

        # validate or make a new output buffer
        imout = __GPU_DMAS_Beamformer__.__check_or_init_buffer__(self, buffer)

        # get the dmassum flag set for signed-square root or power dmas
        if   self.dmastype ==   "ssr": dmasflag = np.int32(0)
        elif self.dmastype == "power": dmasflag = np.int32(1)

        # get the sumlags flag to determine whether to sum over the correlation axis
        if self.sumlags: sumlagflag = np.int32(1)
        else:            sumlagflag = np.int32(0)

        # calculate DMAS
        params = __BMFRM_PARAMS__[self.id]
        kernel_params = (
            imbf,
            np.int32(self.nrx),
            np.int32(self.nop),
            params['lags'],
            np.int32(self.nlags),
            dmasflag,
            sumlagflag,
            imout
        )

        nblock = int(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))
        calc_dmas((nblock,1,1), (self.nthread,1,1), kernel_params)
        
        # convert to numpy or return as cupy array
        if out_as_numpy: return cp.asnumpy(imout)
        else: return imout

@dataclass(kw_only=True)
class TabbedDMASBeamformer(TabbedBeamformer, __GPU_DMAS_Beamformer__):
    """Delay multiply and sum beamformer using arrays of pre-computed delay tabs"""

    def __post_init__(self):
        import cupy as cp
        __GPU_DMAS_Beamformer__.__post_init__(self)
        TabbedBeamformer.__post_init__(self)

        params = __BMFRM_PARAMS__[self.id]

        params['lags'] = cp.ascontiguousarray(cp.array(self.lags), dtype=cp.int32)

        __BMFRM_PARAMS__[self.id] = params

    def __run_base_beamforming__(self, txrxt):
        return TabbedBeamformer.__call__(self, txrxt=txrxt, out_as_numpy=False)

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:
        return __GPU_DMAS_Beamformer__.__call__(self, txrxt, out_as_numpy, buffer)
    
@dataclass(kw_only=True)
class SyntheticDMASBeamformer(SyntheticBeamformer, __GPU_DMAS_Beamformer__):
    """Delay multiply and sum beamformer using arrays of pre-computed delay tabs"""

    def __post_init__(self):
        import cupy as cp
        __GPU_DMAS_Beamformer__.__post_init__(self)
        SyntheticBeamformer.__post_init__(self)

        params = __BMFRM_PARAMS__[self.id]

        params['lags'] = cp.ascontiguousarray(cp.array(self.lags), dtype=cp.int32)

        __BMFRM_PARAMS__[self.id] = params

    def __run_base_beamforming__(self, txrxt):
        return SyntheticBeamformer.__call__(self, txrxt=txrxt, out_as_numpy=False)

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:
        return __GPU_DMAS_Beamformer__.__call__(self, txrxt, out_as_numpy, buffer)