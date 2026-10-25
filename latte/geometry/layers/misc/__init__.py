from .LModules import LorentzAct, QuickDirtyMaxPool, LorentzMaxPool1D, LorentzAvgPool1d, LorentzCentroidMaxPool1D, LorentzSoftMaxPool1D
from .HypLoRAAdapters import GeodesicLoRAAdapter, HypLoRAAdapter

LORENTZ_POOL1D_TYPES = {
    "dirty": QuickDirtyMaxPool,   # per-coordinate max on the space part, then lift
    "clean": LorentzMaxPool1D,    # point farthest from the origin in each window
    "clean_centroid": LorentzCentroidMaxPool1D,
    "soft": LorentzSoftMaxPool1D,
    "average": LorentzAvgPool1d,
}
