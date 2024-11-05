from dataclasses import dataclass, field
from typing import ClassVar
from numpy import ndarray
import logging
logger = logging.getLogger(__name__)

global __BMFRM_PARAMS__
__BMFRM_PARAMS__ = {}

class BeamformerException(Exception): pass

@dataclass(kw_only=True)
class Tabbed():    
    tautx  : ndarray = field(init=True)
    taurx  : ndarray = field(init=True)
    apodtx : ndarray = field(init=True)
    apodrx : ndarray = field(init=True)
    ntx : int = field(init=False)
    nrx : int = field(init=False)
    nop : int = field(init=False)

    def __post_init__(self):
        from numpy import ndim

        # check that all inputs are 2D
        all2d = (ndim(self.tautx) == 2) and (ndim(self.taurx) == 2) and (ndim(self.apodtx) == 2) and (ndim(self.apodrx) == 2)

        # extract dimensions of tx/rx/output point array
        ntx = self.tautx.shape[0]
        nrx = self.taurx.shape[0]
        nop = self.tautx.shape[1]

        # ensure dimensions are consistent
        alltxdim = (self.apodtx.shape[0] == ntx)
        allrxdim = (self.apodrx.shape[0] == nrx)
        allopdim = (self.taurx.shape[1] == nop) and (self.apodtx.shape[1] == nop) and (self.apodtx.shape[1] == nop)

        if not all2d:
            raise BeamformerException(f"all delay tabs and apodizations must be 2D matrices")
        if not alltxdim:
            raise BeamformerException(f"tautx and apodtx must have the same shape in dim 0 but were {self.tautx.shape[0]} and {self.apodtx.shape[0]}")
        if not allrxdim:
            raise BeamformerException(f"taurx and apodrx must have the same shape in dim 0 but were {self.taurx.shape[0]} and {self.apodrx.shape[0]}")
        if not allopdim:
            raise BeamformerException(f"tautx, taurx, apodtx, and apodrx must have the same shape in dim 1 but were {self.tautx.shape[0]}, {self.taurx.shape[0]}, {self.apodtx.shape[0]}, and {self.apodrx.shape[0]}")
        
        self.ntx = ntx
        self.nrx = nrx
        self.nop = nop

@dataclass(kw_only=True)
class Parallelized():
    nbf : ClassVar[int] = 0
    id  : int = field(init=False)

    def __post_init__(self):
        self.id = Parallelized.nbf
        Parallelized.nbf += 1

        __BMFRM_PARAMS__[self.id] = {}