import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from .utils import CacheTensor
import torch

class LibriSpeech2MixDataset(Dataset):
    def __init__(
            self, 
            df: pd.DataFrame, 
            sample_rate:int = 16000,
            using_cache = False,
            cache_size = 1,
            group_label = True
            ):
        super().__init__()
        self.data = df
        self.sample_rate = sample_rate
        self.group_label = group_label
        if not using_cache or cache_size == 1:
            self.file_source = torchaudio.load
        else: 
            self.file_source = CacheTensor(cache_size,torchaudio.load)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        audio_file = data['audio_file']
        from_idx = data['from_idx']
        to_idx = data['to_idx']
        mix_audio_file = data['mix_audio_file']
        mix_from_idx = data['mix_from_idx']
        mix_to_idx = data['mix_to_idx']

        first_waveform,rate = self.file_source(audio_file)
        if rate != self.sample_rate:
            first_waveform = torchaudio.functional.resample(first_waveform,rate,self.sample_rate)
        second_waveform,rate = self.file_source(mix_audio_file)
        first_waveform = first_waveform.squeeze()[from_idx:to_idx]
        if rate != self.sample_rate:
            second_waveform = torchaudio.functional.resample(second_waveform,rate,self.sample_rate)
        second_waveform = second_waveform.squeeze()[mix_from_idx:mix_to_idx]
        mix_waveform = torchaudio.functional.add_noise(first_waveform,second_waveform,torch.tensor([1]))

        return mix_waveform, [first_waveform, second_waveform] if self.group_label else mix_waveform, first_waveform, second_waveform


class LibriSpeech2MixWithSpeakerSampleDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
