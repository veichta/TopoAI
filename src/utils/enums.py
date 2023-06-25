from enum import Enum


class DatasetEnum(Enum):
    CIL = "cil"
    EPFL = "epfl"
    ROADTRACER = "roadtracer"
    DEEPGLOBE = "deepglobe"
    MIT = "mit"
    ALL = "all"

class ModelsEnum(Enum):
    UNET = "unet"
    UNETPP = "unet++"
    