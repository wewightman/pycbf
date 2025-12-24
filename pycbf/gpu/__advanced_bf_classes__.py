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

    def __post_init__(self):
        TabbedBeamformer.__post_init__(self)

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray: pass