import torch.optim as optim

BASE_OPTIMIZER = optim.AdamW
BASE_OPTIMIZER_CONFIG = {
    "lr": 1.0e-3,
    "weight_decay": 1.0e-2
}