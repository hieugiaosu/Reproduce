from dataclasses import dataclass
from abc import ABC
from typing import Optional

class ModelConfig(ABC):
    def toDict(self) -> dict:
        classDict = {}
        for attribute_name in dir(self):
            if not attribute_name.startswith("__") and not callable(getattr(self, attribute_name)) and not attribute_name.startswith("_"):
                attribute_value = getattr(self, attribute_name)
                if isinstance(attribute_value, ModelConfig):
                    classDict[attribute_name] = attribute_value.toDict()
                else:
                    classDict[attribute_name] = attribute_value
        return classDict


@dataclass
class ModuleAudioEncConfig(ModelConfig):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    groups: int
    bias: bool

@dataclass
class ModuleFeatureProjectorConfig(ModelConfig):
    num_channels: int
    in_channels: int
    out_channels: int
    kernel_size: int
    bias: bool

@dataclass
class RelativePositionalEncodingConfig(ModelConfig):
    in_channels: int
    num_heads: int
    maxlen: int
    embed_v: bool

@dataclass
class GlobalBlocksConfig(ModelConfig):
    in_channels: int
    num_mha_heads: int
    dropout_rate: float

@dataclass
class LocalBlocksConfig(ModelConfig):
    in_channels: int
    kernel_size: int
    dropout_rate: float

@dataclass
class DownConvLayerConfig(ModelConfig):
    in_channels: int
    samp_kernel_size: int

@dataclass
class EncStageConfig(ModelConfig):
    global_blocks: GlobalBlocksConfig
    local_blocks: LocalBlocksConfig
    down_conv_layer: DownConvLayerConfig

@dataclass
class SpkSplitStageConfig(ModelConfig):
    in_channels: int
    num_spks: int

@dataclass
class SimpleFusionConfig(ModelConfig):
    out_channels: int

@dataclass
class SpkAttentionConfig(ModelConfig):
    in_channels: int
    num_mha_heads: int
    dropout_rate: float

@dataclass
class DecStageConfig(ModelConfig):
    num_spks: int
    global_blocks: GlobalBlocksConfig
    local_blocks: LocalBlocksConfig
    spk_attention: SpkAttentionConfig

@dataclass
class ModuleSeparatorConfig(ModelConfig):
    num_stages: int
    relative_positional_encoding: RelativePositionalEncodingConfig
    enc_stage: EncStageConfig
    spk_split_stage: SpkSplitStageConfig
    simple_fusion: SimpleFusionConfig
    dec_stage: DecStageConfig

@dataclass
class ModuleOutputLayerConfig(ModelConfig):
    in_channels: int
    out_channels: int
    num_spks: int

@dataclass
class ModuleAudioDecConfig(ModelConfig):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    bias: bool

@dataclass
class SepReformerConfig(ModelConfig):
    num_stages: int
    num_spks: int
    module_audio_enc: ModuleAudioEncConfig
    module_feature_projector: ModuleFeatureProjectorConfig
    module_separator: ModuleSeparatorConfig
    module_output_layer: ModuleOutputLayerConfig
    module_audio_dec: ModuleAudioDecConfig