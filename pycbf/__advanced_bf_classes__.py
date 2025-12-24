from pycbf.__bf_base_classes__ import Beamformer, BeamformerException
from dataclasses import dataclass, field
from typing import Literal

from numpy import ndarray as npNDArray

@dataclass(kw_only=True)
class DMASBeamformer():
    """Delay multiply and sum beamformer base class"""
    lags : npNDArray = field(init=True)
    lagaxis : Literal["tx", "rx"] = "rx"
    nlags : int = field(init=False)
    sumlags : bool = True
    sumtype : str = field(init=False)
    dmastype : Literal["ssr", "power"] = "ssr"

    def __post_init__(self):
        # based on the lag/correlation axis, set the sumtype for the DAS beamformer
        if   self.lagaxis == "rx": self.sumtype = "tx_only"
        elif self.lagaxis == "tx": self.sumtype = "rx_only"
        else: raise BeamformerException("lagaxis keyword can only be 'tx' or 'rx'")

        if (self.dmastype != "ssr") and (self.dmastype != "power"): raise BeamformerException("dmastype keyword must be 'ssr' (signed-square-root) or 'power'")

        self.nlags = len(self.lags)
