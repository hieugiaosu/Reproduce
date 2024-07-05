from .loss_function import *

BASE_LOSS_CONFIG = {
  "PIT_SISNR_mag": {
    "frame_length": 512,
    "frame_shift": 128,
    "window": "hann",
    "num_stages": "var_model_num_stages",
    "num_spks": 2,
    "scale_inv": True,
    "mel_opt": False
  },
  "PIT_SISNR_time": {
    "num_spks": 2,
    "scale_inv": True
  },
  "PIT_SISNRi": {
    "num_spks": 2,
    "scale_inv": True
  },
  "PIT_SDRi": {
    "dump": 0
  }
}