from abc import ABC, abstractmethod

global __BMFRM_DEBUG__
__BMFRM_DEBUG__ = False

global __BMFRM_PARAMS__
__BMFRM_PARAMS__ = []

class Beamformer(ABC):
    __bmfrm_id_counter__ = int(0)

    def __init__(self):
        self.id = int(Beamformer.__bmfrm_id_counter__)
        __BMFRM_PARAMS__.append(dict())
        Beamformer.__bmfrm_id_counter__ += int(1)

    @abstractmethod
    def __init_tabs__(self):
        """Generate the delay tabs"""
        pass

    @abstractmethod
    def __init_masks__(self):
        """Generate the delay tabs"""
        pass