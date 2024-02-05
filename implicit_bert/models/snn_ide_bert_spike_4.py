import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import copy
sys.path.append("../")
from implicit_bert.modules.snnide_bert_multilayer_module import SNNIDEBERTSpikingMultiLayerModule
from implicit_bert.modules.snn_modules import SNNFC, SNNBERTSpikingLIFFuncMultiLayer
from implicit_bert.modules.snn_bert_modules import BertConfig, BertSelfAttentionSplit, BertSelfOutput, BertIntermediate, BertOutput, BertEmbeddings, BertLayerNorm, BertFC
logger = logging.getLogger(__name__)

# Network for 4 Spiking Encoder layers
class SNNIDESTUDENTBERT4Spike(nn.Module):

    def __init__(self, cfg_path, load_pretrained = False, t_conv = 100, vth = 1., **kwargs):
        super(SNNIDESTUDENTBERT4Spike, self).__init__()

        # Hyperparameters
        self.threshold = 30
        self.time_step = t_conv
        self.vth = vth
        self.dropout = 0.1
        self.leaky = 1. #1 means IF neuron; 0<self.leaky<1 mean LIF
        self.solver = 'broy'

        config = BertConfig.from_json_file(cfg_path + '/config.json')
        self.config = config

        self.network_x = BertEmbeddings(config)
        # Input to all layers are spikes
        self.network_s1 = BertFC(config)
        self.network_s2 = BertFC(config)
        self.network_s3 = BertSelfAttentionSplit(config)
        self.network_s4 = BertSelfOutput(config)
        self.network_s5 = BertIntermediate(config)
        self.network_s6 = BertOutput(config)

        self.network_s7 = BertFC(config)
        self.network_s8 = BertFC(config)
        self.network_s9 = BertSelfAttentionSplit(config)
        self.network_s10 = BertSelfOutput(config)
        self.network_s11 = BertIntermediate(config)
        self.network_s12 = BertOutput(config)
        self.network_s13 = BertFC(config)
        self.network_s14 = BertFC(config)
        self.network_s15 = BertSelfAttentionSplit(config)
        self.network_s16 = BertSelfOutput(config)
        self.network_s17 = BertIntermediate(config)
        self.network_s18 = BertOutput(config)
        self.network_s19 = BertFC(config)
        self.network_s20 = BertFC(config)
        self.network_s21 = BertSelfAttentionSplit(config)
        self.network_s22 = BertSelfOutput(config)
        self.network_s23 = BertIntermediate(config)
        self.network_s24 = BertOutput(config)
        # Feedback: optional
        self.network_s25 = SNNFC(config.hidden_size,config.hidden_size)

        self.snn_func = SNNBERTSpikingLIFFuncMultiLayer(nn.ModuleList([self.network_s1, self.network_s2, self.network_s3, self.network_s4, self.network_s5, self.network_s6, self.network_s7, self.network_s8, self.network_s9, self.network_s10, self.network_s11, self.network_s12, self.network_s13, self.network_s14, self.network_s15, self.network_s16, self.network_s17, self.network_s18, self.network_s19, self.network_s20, self.network_s21, self.network_s22, self.network_s23, self.network_s24, self.network_s25]), self.network_x, vth=self.vth, leaky=self.leaky)

        self.snn_func_copy = copy.deepcopy(self.snn_func)

        for param in self.snn_func_copy.parameters():
            param.requires_grad_(False)
        self.fit_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.snn_ide_conv = SNNIDEBERTSpikingMultiLayerModule(self.snn_func, self.snn_func_copy)
        if load_pretrained:
            self.from_pretrained(cfg_path + '/pytorch_model.bin')
        else:
            self.apply(self.init_bert_weights)
            #self.fit_dense.weight.data.normal_(
            #    mean=0.0, std=self.config.initializer_range)
            #self.fit_dense.bias.data.zero_()
    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        self.load_state_dict(state_dict=state_dict)
        return

    def _forward(self, input_ids, segment_ids, attention_mask, **kwargs):
        threshold = kwargs.get('threshold', self.threshold)
        time_step = kwargs.get('time_step', self.time_step)
        input_type = kwargs.get('input_type', 'constant')
        leaky = kwargs.get('leaky', self.leaky)

        student_rep, attn_vals = self.snn_ide_conv(input_ids, segment_ids, attention_mask = attention_mask, time_step=time_step, threshold=threshold, input_type=input_type, solver_type=self.solver, leaky=leaky)

        return student_rep, attn_vals

    def forward(self, input_ids, segment_ids, attention_mask, **kwargs):
        student_rep, attn_vals = self._forward(input_ids, segment_ids, attention_mask, **kwargs)
        tmp = []

        for s_id, sequence_layer in enumerate(student_rep):
            tmp.append(self.fit_dense(sequence_layer))
        student_rep = tmp
        return student_rep, attn_vals


