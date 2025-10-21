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
class Synthetic():    
    ovectx  : ndarray = field(init=True)    # source point location (ntx by ndim matrix)
    nvectx  : ndarray = field(init=True)    # normal vector of the source point (ntx by ndim matrix)
    doftx   : ndarray = field(init=True)    # depth of field over which to use planar delay tabs around ovec (ntx length vector)
    alatx   : ndarray = field(init=True)    # angular acceptance around nvec (fnumber basically) (ntx length vector)
    t0tx    : ndarray = field(init=True)    # time the wave front passes through ovectx (ntx length vector)

    ovecrx  : ndarray = field(init=True)    # location of the recieve sensors (nrx by ndim matrix)
    nvecrx  : ndarray = field(init=True)    # orientation of the recieve sensor (nrx by ndim matrix)
    alarx   : ndarray = field(init=True)    # acceptance angle of the recieve sensor (nrx length vector)

    pnts    : ndarray = field(init=True)    # number of reconstruction points (nop by ndim matrix)

    ntx     : int = field(init=False)       # number of transmit events
    nrx     : int = field(init=False)       # number of recieve events
    nop     : int = field(init=False)       # number of points to recon
    ndimp   : int = field(init=False)       # number of spatial dimensions to reconstruct over (2 or 3)

    def __post_init__(self):
        from numpy import ndim

        # check that all vector inputs are 1D
        all1d = (ndim(self.doftx) == 1) and (ndim(self.alatx) == 1) and (ndim(self.t0tx) == 1) and (ndim(self.alarx) == 1) 

        # check that all matrix inputs are 2D
        all2d = (ndim(self.ovectx) == 2) and (ndim(self.nvectx) == 2) and (ndim(self.ovecrx) == 2) and (ndim(self.nvecrx) == 2)

        # extract dimensions of tx/rx/output point array
        ntx     = self.ovectx.shape[0]
        nrx     = self.ovecrx.shape[0]
        nop     = self.pnts.shape[0]
        ndimp   = self.pnts.shape[1]

        # ensure dimensions are consistent
        alltxdim = (self.nvectx.shape[0] == ntx) and (self.doftx.shape[0] == ntx) and (self.alatx.shape[0] == ntx) and (self.t0tx.shape[0] == ntx)
        allrxdim = (self.nvecrx.shape[0] == nrx) and (self.alarx.shape[0] == nrx)
        allndimp = (self.ovectx.shape[1] == ndimp) and (self.nvectx.shape[1] == ndimp) and (self.ovecrx.shape[1] == ndimp) and (self.nvecrx.shape[1] == ndimp)

        if not all1d:
            raise BeamformerException(f"all acceptance angle, depth of field, and t0 vectors must be 1D")
        if not all2d:
            raise BeamformerException(f"all origin and normal vector matrices must be 2D")
        if not alltxdim:
            raise BeamformerException(f"All tx variables must have the same shape in the first dimension")
        if not allrxdim:
            raise BeamformerException(f"All rx variables must have the same shame in the first variable")
        if not allndimp:
            raise BeamformerException(f"All matrix variables must have the same number of dimensions")
        
        self.ntx    = ntx
        self.nrx    = nrx
        self.nop    = nop
        self.ndimp  = ndimp

@dataclass(kw_only=True)
class Parallelized():
    nbf : ClassVar[int] = 0
    id  : int = field(init=False)

    def __post_init__(self):
        self.id = Parallelized.nbf
        Parallelized.nbf += 1

        __BMFRM_PARAMS__[self.id] = {}

@dataclass(kw_only=True)
class InterpAndSumTyped():
    interp : dict = field(init=True)
    sumtype : str = 'tx_and_rx'

    def __post_init__(self):
        self.__check_interp__()
        self.__check_sumtype__()
    
    def __check_sumtype__(self):
        valid_keys = ['none', 'tx_only', 'rx_only', 'tx_and_rx']

        err_msg = f"sumtype was {self.sumtype} but must be one of the following: "

        for key in valid_keys: err_msg += key + ", "
        err_msg = err_msg[:-2]
        
        if self.sumtype not in valid_keys: raise BeamformerException(err_msg)

    def __check_interp__(self):
        valid_kinds = ["nearest", "linear", "akima", "makima", "korder_cubic"]
        kind = self.interp.get('kind', None)
        if (kind is None) or (kind not in valid_kinds): 
            err_msg = f"interp['kind'] was {kind} but must be one of the following: "

            for key in valid_kinds: err_msg += key + ", " 
            err_msg = err_msg[:-2]

            raise BeamformerException(err_msg)
        
        if kind == 'nearest':
            usf = self.interp.get("usf", None)
            if usf is None: self.interp['usf'] = 1
            elif (usf < 1): raise BeamformerException("'usf' for interp kind 'nearest' must be an interger greater than 1")
        elif kind == 'korder_cubic':
            k = self.interp.get("k", None)
            if k is None: self.interp['k'] = 4
            if (k < 4) or (k%2 != 0): raise BeamformerException("'k' for interp kind 'korder_cubic' must be at least four and even")
