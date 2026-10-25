from .LFC import LorentzFullyConnectedChen, lorentz_init_
from .LPLFC import LorentzPLFCCustom, LorentzPLFCScaled
from .LFCLora import LorentzFullyConnectedBoostLora, LorentzFullyConnectedLora

# linear_type -> Lorentz fully-connected layer
LORENTZ_FC_TYPES = {"base": LorentzFullyConnectedChen,
                    "chen": LorentzFullyConnectedChen,
                    "plfc": LorentzPLFCCustom,
                    "plfc_scaled": LorentzPLFCScaled}
