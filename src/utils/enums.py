from enum import Enum


class DatasetEnum(Enum):
    CIL = "cil"
    EPFL = "epfl"
    ROADTRACER = "roadtracer"
    ALL = "all"

class ModelsEnum(Enum):
    UNET = "unet"
    UNETPP = "unet++"
    SPIN = "spin"
    UperNet_T = "upernet-t"
    UperNet_B = "upernet-b"
    UperNet_L = "upernet-l"
    