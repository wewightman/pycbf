from pycbf.__bf_base_classes__ import __BMFRM_PARAMS__
from pycbf.gpu.__core_bf_classes__ import TabbedBeamformer, SyntheticBeamformer
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from numpy import ndarray as npNDArray
from cupy  import ndarray as cpNDArray

@dataclass(kw_only=True)
class TabbedDMASBeamformer(TabbedBeamformer):
    """Delay multiply and sum beamformer using arrays of pre-computed delay tabs"""
    lags : npNDArray = field(init=True)
    lagaxis : Literal["tx", "rx"] = "rx"
    nlags : int = field(init=False)
    sumlags : bool = True
    sumtype : str = field(init=False)
    dmastype : Literal["ssr", "power"] = "ssr"
    dmasfilt : dict = field(init=True,default_factory=lambda:{"kind":"none"})

    def __post_init__(self):
        import numpy as np
        import cupy as cp

        # based on the lag/correlation axis, set the sumtype for the DAS beamformer
        if   self.lagaxis == "rx": self.sumtype = "tx_only"
        elif self.lagaxis == "tx": self.sumtype = "rx_only"

        TabbedBeamformer.__post_init__(self)

        self.lags = np.ascontiguousarray(self.lags, dtype=int)
        self.nlags = len(self.lags)
        params = __BMFRM_PARAMS__[self.id]

        params['lags'] = cp.ascontiguousarray(cp.array(self.lags), dtype=cp.int32)

        __BMFRM_PARAMS__[self.id] = params

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
        imbf = TabbedBeamformer.__call__(self, txrxt=txrxt, out_as_numpy=False)

        # validate or make a new output buffer
        imout = self.__check_or_init_buffer__(buffer)

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