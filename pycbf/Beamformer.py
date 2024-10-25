from dataclasses import dataclass
from numpy import ndarray
import logging
logger = logging.getLogger(__name__)

class BeamformerException(Exception): pass

@dataclass(kw_only=True)
class Beamformer():
    tautx  : ndarray
    taurx  : ndarray
    apodtx : ndarray
    apodrx : ndarray

    def __post_init__(self):
        if self.tautx.shape[0] != self.apodtx.shape[0]:
            raise BeamformerException(f"tautx and apodtx must have the same shape in dim 0 but were {self.tautx.shape[0]} and {self.apodtx.shape[0]}")
        self.ntx = self.tautx.shape[0]
