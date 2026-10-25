# GyroAtt baseline

fp32 port of GyroAtt (Wang et al., NeurIPS 2025; github.com/ChenHu-ML/GyroAtt),
SPD manifold with the log-Euclidean metric (GyroAtt-SPD-LEM). Table 5 builds it
through `baselines/manifold/gyroatt_arm.py` and runs it with
`baselines/manifold/moabb_protocol.py --model gyroatt [--input_align]`.

| File | Contents |
|---|---|
| `models.py` | `GyroAttBCI` (motor imagery) and `GyroAttMamem`, the Euclidean-to-SPD front end (`Signal2Spd`, `E2R`) and the `GyroAttention` dispatcher |
| `spd_gyro.py` | SPD gyro-attention under AIM / LEM / LCM |
| `grassmann_gyro.py` | Grassmann gyro-attention, used only by the PSPD composition |
| `functional.py` | eigendecomposition-based matrix functions and Grassmann helpers |
| `conf/gyroatt_bci.yaml` | architecture and optimiser values |

