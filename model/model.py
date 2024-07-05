import torch.nn as nn
from .component import *
from .config import SepReformerConfig

class SepReformer(nn.Module):
    def __init__(self, config: SepReformerConfig|None = None, config_in_dict_type:dict|None=None):
        '''
        config parameter is for type safe, if config is None, the **args will be a parameter to initialize
        the SepReformer
        '''

        super().__init__()
        if config is not None: 
            config_dict = config.toDict()
            num_stages = config_dict['num_stages']
            num_spks = config_dict['num_spks']
            module_audio_enc = config_dict['module_audio_enc']
            module_feature_projector = config_dict['module_feature_projector']
            module_separator = config_dict['module_separator']
            module_output_layer = config_dict['module_output_layer']
            module_audio_dec = config_dict['module_audio_dec']
        else: 
            num_stages = config_in_dict_type['num_stages']
            num_spks = config_in_dict_type['num_spks']
            module_audio_enc = config_in_dict_type['module_audio_enc']
            module_feature_projector = config_in_dict_type['module_feature_projector']
            module_separator = config_in_dict_type['module_separator']
            module_output_layer = config_in_dict_type['module_output_layer']
            module_audio_dec = config_in_dict_type['module_audio_dec']

        self.num_stages = num_stages
        self.num_spks = num_spks
        self.audio_encoder = AudioEncoder(**module_audio_enc)
        self.feature_projector = FeatureProjector(**module_feature_projector)
        self.separator = Separator(**module_separator)
        self.out_layer = OutputLayer(**module_output_layer)
        self.audio_decoder = AudioDecoder(**module_audio_dec)
        
        # Aux_loss
        self.out_layer_bn = nn.ModuleList([])
        self.decoder_bn = nn.ModuleList([])
        for _ in range(self.num_stages):
            self.out_layer_bn.append(OutputLayer(**{**config.module_output_layer.__dict__, "masking": True}))
            self.decoder_bn.append(AudioDecoder(**config.module_audio_dec.__dict__))
        
    def forward(self, x):
        encoder_output = self.audio_encoder(x)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)
        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        each_spk_output = [out_layer_output[idx] for idx in range(self.num_spks)]
        audio = [self.audio_decoder(each_spk_output[idx]) for idx in range(self.num_spks)]
        
        # Aux_loss
        audio_aux = []
        for idx, each_stage_output in enumerate(each_stage_outputs):
            each_stage_output = self.out_layer_bn[idx](nn.functional.upsample(each_stage_output, encoder_output.shape[-1]), encoder_output)
            out_aux = [each_stage_output[jdx] for jdx in range(self.num_spks)]
            audio_aux.append([self.decoder_bn[idx](out_aux[jdx])[...,:x.shape[-1]] for jdx in range(self.num_spks)])
            
        return audio, audio_aux