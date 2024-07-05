from abc import ABC
import torch

class TrainPipeline(ABC):
    def __init__(
            self,
            model,
            optimizer,
            dataloader,
            device = "cpu"
            ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
    def train(self,epoch,time_limit,**kwargs):
        pass

def get_all_cuda_device_ids():
    device_ids = []
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_ids = list(range(num_devices))
    return device_ids

def check_device_is_cuda(device: torch.device):
    return device.type == 'cuda'