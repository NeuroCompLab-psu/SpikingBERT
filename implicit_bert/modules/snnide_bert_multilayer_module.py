# Identifiers will be add once the code is made public.

import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from implicit_bert.modules.broyden import broyden
import logging
logger = logging.getLogger(__name__)


class SNNIDEBERTSpikingMultiLayerModule(nn.Module):
    """
    SNN module with implicit differentiation on the equilibrium point in the inner 'Backward' class.
    """

    def __init__(self, snn_func, snn_func_copy):
        super(SNNIDEBERTSpikingMultiLayerModule, self).__init__()
        self.snn_func = snn_func
        self.snn_func_copy = snn_func_copy

    def forward(self, u, segment_ids, **kwargs):
        time_step = kwargs.get('time_step', 30)
        threshold = kwargs.get('threshold', 30)
        input_type = kwargs.get('input_type', 'constant')
        solver_type = kwargs.get('solver_type', 'broy')
        leaky = kwargs.get('leaky', None)
        get_all_rate = kwargs.get('get_all_rate', False)
        attention_mask = kwargs.get('attention_mask', None)
        with torch.no_grad():
            if attention_mask is None:
                attention_mask = torch.ones_like(u)
            if segment_ids is None:
                segment_ids = torch.zeros_like(u)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Process attention mask
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            # Gets the ASR of layers required for computing loss during KD and final training in z_layers
            z_layers, attn_avg, first_attn_vals = self.snn_func.snn_forward(u, segment_ids, time_step, input_type=input_type, attention_mask=extended_attention_mask, output_type='all_layers')

        if self.training:

            # For embedding + 4 encoders (embedding, SE1, SE2, SE3, SE4)
            z0_out_ = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-25], u, segment_ids, num = 0, attention_mask=extended_attention_mask) #-13
            z3_out_ = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-19], u, segment_ids, num = 6, attention_mask=extended_attention_mask) #-7
            z6_out_ = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-13], u, segment_ids, num = 12, attention_mask=extended_attention_mask)
            z9_out_ = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-7], u, segment_ids, num = 18, attention_mask=extended_attention_mask)
            z12_out_ = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-1], u, segment_ids, num = 24, attention_mask=extended_attention_mask)

            self.snn_func_copy.copy(self.snn_func)

            attn_score_1 = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-1], u, segment_ids, num=24,
                                                                    attention_mask=extended_attention_mask, is_attn=1)
            attn_score_2 = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-1], u, segment_ids, num=24, #-13,0
                                                                    attention_mask=extended_attention_mask, is_attn=2)
            attn_score_3 = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-1], u, segment_ids, num=24,
                                                                    attention_mask=extended_attention_mask, is_attn=3)
            attn_score_4 = self.snn_func.equivalent_func_per_layer_bert_spiking_specific(z_layers[-1], u, segment_ids, num=24,
                                                                    attention_mask=extended_attention_mask, is_attn=4)

            # All sizes are same
            sizes_last = z0_out_.size()
            B = z0_out_.size(0)
            z0_out_ = z0_out_.reshape(B, -1, 1)

            sizes_last = z3_out_.size()
            B = z3_out_.size(0)
            z3_out_ = z3_out_.reshape(B, -1, 1)

            sizes_first = z6_out_.size()
            B = z6_out_.size(0)
            z6_out_ = z6_out_.reshape(B, -1, 1)

            sizes_first = z9_out_.size()
            B = z9_out_.size(0)
            z9_out_ = z9_out_.reshape(B, -1, 1)

            sizes_first = z12_out_.size()
            B = z12_out_.size(0)
            z12_out_ = z12_out_.reshape(B, -1, 1)


            # If threshold < 0 we just take initial estimate from grad which works fine for no-feedback case.
            if threshold < 0:
                layer_num = 0
                z0_out_ = self.Backward.apply(self.snn_func_copy, z0_out_, u, segment_ids,  layer_num, extended_attention_mask, sizes_last, threshold, solver_type)
                layer_num = 6
                z3_out_ = self.Backward.apply(self.snn_func_copy, z3_out_, u, segment_ids,  layer_num, extended_attention_mask, sizes_last, threshold, solver_type)
                layer_num = 12
                z6_out_ = self.Backward.apply(self.snn_func_copy, z6_out_, u, segment_ids, layer_num, extended_attention_mask, sizes_first, threshold, solver_type)
                layer_num = 18
                z9_out_ = self.Backward.apply(self.snn_func_copy, z9_out_, u, segment_ids, layer_num, extended_attention_mask, sizes_first, threshold, solver_type)
                layer_num = 24
                z12_out_ = self.Backward.apply(self.snn_func_copy, z12_out_, u, segment_ids, layer_num, extended_attention_mask, sizes_first, threshold, solver_type)
            # change back the dimension
            z0_out_ = torch.reshape(z0_out_, sizes_last)
            z3_out_ = torch.reshape(z3_out_, sizes_last)
            z6_out_ = torch.reshape(z6_out_, sizes_first)
            z9_out_ = torch.reshape(z9_out_, sizes_first)
            z12_out_ = torch.reshape(z12_out_, sizes_first)

            z0_out = self.Replace.apply(z0_out_, z_layers[-25])
            z3_out = self.Replace.apply(z3_out_, z_layers[-19])
            z6_out = self.Replace.apply(z6_out_, z_layers[-13])
            z9_out = self.Replace.apply(z9_out_, z_layers[-7])
            z12_out = self.Replace.apply(z12_out_, z_layers[-1])


        else:
            # During Testing ASR directly from the neurons and no additional steps
            z0_out = z_layers[-25]
            z3_out = z_layers[-19]
            z6_out = z_layers[-13]
            z9_out = z_layers[-7]
            z12_out = z_layers[-1]

            # No use of these values during inference, only z12_out is required i.e. ASR of final SE layer
            attn_score_1 = attn_avg[0]
            attn_score_2 = attn_avg[1]
            attn_score_3 = attn_avg[2]
            attn_score_4 = attn_avg[3]

        return [z0_out, z3_out, z6_out, z9_out, z12_out], [attn_score_1, attn_score_2, attn_score_3, attn_score_4] #, z9_out, z12_out]

    class Replace(Function):
        @staticmethod
        def forward(ctx, z1, z1_r):
            return z1_r

        @staticmethod
        def backward(ctx, grad):
            return (grad, grad)

    class Backward(Function):
        @staticmethod
        def forward(ctx, snn_func_copy, z1, u, segment_ids, layer_num, attention_mask, *args):
            ctx.save_for_backward(z1)
            ctx.u = u
            ctx.segment_ids = segment_ids
            ctx.snn_func = snn_func_copy
            ctx.args = args
            ctx.layer_num = layer_num
            ctx.attention_mask = attention_mask

            return z1.clone()

        @staticmethod
        def backward(ctx, grad):
            # torch.cuda.empty_cache()
            #pydevd.settrace(suspend=False, trace_only_current_thread=True)
            # grad should have dimension (bsz x d_model x seq_len) to be consistent with the solver
            bsz, d_model, seq_len = grad.size()
            grad = grad.clone()
            z1, = ctx.saved_tensors
            u = ctx.u
            segment_ids = ctx.segment_ids
            layer_num = ctx.layer_num
            attention_mask = ctx.attention_mask
            #print('backward : ', layer_num)

            args = ctx.args
            sizes, threshold, solver_type = args[-3:]

            snn_func = ctx.snn_func
            z1_temp = z1.clone().detach().requires_grad_()
            u_temp = u.clone().detach()
            segment_ids_temp = segment_ids.clone().detach()
            #print('Backward is layer: ', layer_num)
            def infer_from_vec(z, u, segment_ids):
                # change the dimension of z
                B = sizes[0]
                z_in = torch.reshape(z, sizes)

                return (snn_func.equivalent_func_per_layer_bert_spiking_specific(z_in, u, segment_ids, num = layer_num, attention_mask=attention_mask) - z_in).reshape(B, -1, 1)

            with torch.enable_grad():
                y = infer_from_vec(z1_temp, u_temp, segment_ids_temp)

            def g(x):
                y.backward(x, retain_graph=True)  # Retain for future calls to g
                res = z1_temp.grad.clone().detach() + grad
                z1_temp.grad.zero_()
                return res

            if solver_type == 'broy':
                # print('Backward is used! ', args.position)
                eps = 2e-10 * np.sqrt(bsz * seq_len * d_model) # previously 2
                dl_df_est = torch.zeros_like(grad)

                result_info = broyden(g, dl_df_est, threshold=threshold, eps=eps, name="backward")
                dl_df_est = result_info['result']
                nstep = result_info['nstep']
                lowest_step = result_info['lowest_step']
                # print('Layer : ' , layer_num)
                print('NSTEP : ' , nstep)
            else:
                dl_df_est = grad
                for i in range(threshold):
                    dl_df_est = (dl_df_est + g(dl_df_est)) / 2.
                print('Other method!')

            if threshold > 30:
                torch.cuda.empty_cache()

            y.backward(torch.zeros_like(dl_df_est), retain_graph=False)
            #dl_df_est = torch.zeros_like(dl_df_est)
            grad_args = [None for _ in range(len(args)+3)]
            return (None, dl_df_est, None, *grad_args)
