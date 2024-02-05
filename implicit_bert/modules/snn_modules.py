# Identifiers will be add once the code is made public.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
import sys
import os
sys.path.append('../')
from implicit_bert.modules.optimizations import VariationalHidDropout2d, weight_spectral_norm
import random


class SNNFuncMultiLayer(nn.Module):

    def __init__(self, network_s_list, network_x, vth, fb_num=1):
        # network_s_list is a list of networks, the last fb_num ones are the feedback while previous are feed-forward
        super(SNNFuncMultiLayer, self).__init__()
        self.network_s_list = network_s_list
        self.network_x = network_x
        self.vth = torch.tensor(vth, requires_grad=False)
        self.fb_num = fb_num

    def snn_forward(self, x, time_step, output_type='normal', input_type='constant'):
        pass

    # This function is primarily used for training. It leverages the ASR as defined in the paper.
    def equivalent_func_per_layer_bert_spiking_specific(self, a, x, segment_ids, num, attention_mask = None, is_attn = 0, op = 'train'):
        fac = 1.
        min_val = 0
        max_val = 1
        avg_list_prev = []
        avg_list_prev.append(a)
        j = 0
        for i in range(num, len(self.network_s_list) - 1):
            # Comment out this break for feedback
            break
            if i % 6 == 0:
                # Key Layer
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j])) / self.vth, min_val, max_val)
            elif i % 6 == 1:
                # Value Layer
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j-1])) / self.vth, min_val, max_val)
            elif i % 6 == 2:
                a, attn = (self.network_s_list[i](avg_list_prev[j-2], avg_list_prev[j-1], avg_list_prev[j], attention_mask))
                a = torch.clamp((a) / (self.vth), min_val, max_val)
                #if is_attn > 0:
                #    if i + 1 == is_attn:
                #        return attn
            elif i % 6 == 3:
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j], avg_list_prev[j-3])) / self.vth, min_val, max_val)
            elif i % 6 == 5:
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j], avg_list_prev[j-1])) / self.vth, min_val, max_val)
            else:
                a = torch.clamp((self.network_s_list[i](avg_list_prev[j])) / self.vth, min_val, max_val)
            avg_list_prev.append(a)
            j += 1

        # Uncomment this to use feedback
        #a = torch.clamp((self.network_s_list[-1](a) + self.network_x(x, segment_ids)) / self.vth, min_val, max_val)

        # No feedback
        a = torch.clamp((self.network_x(x, segment_ids)) / self.vth, min_val, max_val)

        #a = self.network_s_list[-1](a) + self.network_x(x, segment_ids)
        avg_list = []
        avg_list.append(a)
        if is_attn:
            num = 25
        for i in range(num):
            if i % 6 == 0:
                # Key Layer
                a = torch.clamp((self.network_s_list[i](avg_list[i])) / (self.vth), min_val, max_val)
            elif i % 6 == 1:
                # Value Layer
                a = torch.clamp((self.network_s_list[i](avg_list[i-1])) / (self.vth), min_val, max_val)
            elif i % 6 == 2:
                # Attention layer
                a, attn = (self.network_s_list[i](avg_list[i-2], avg_list[i-1], avg_list[i], attention_mask))
                a = torch.clamp((a) / (self.vth), min_val, max_val)
                if is_attn > 0:
                    if int(i/6) + 1 == is_attn:
                        return attn
            elif i % 6 == 3:
                # IL1
                a = torch.clamp((self.network_s_list[i](avg_list[i], avg_list[i-3])) / self.vth, min_val, max_val)
            elif i % 6 == 5:
                # Output
                a = torch.clamp((self.network_s_list[i](avg_list[i], avg_list[i-1])) / self.vth, min_val, max_val)
            else:
                a = torch.clamp((self.network_s_list[i](avg_list[i])) / self.vth, min_val, max_val)
            avg_list.append(a)
        return a

    def forward(self, x, time_step):
        return self.snn_forward(x, time_step)

    def copy(self, target):
        for i in range(len(self.network_s_list)):
            self.network_s_list[i].copy(target.network_s_list[i])
        self.network_x.copy(target.network_x)


# Spike creation and flow is defined in this class
class SNNBERTSpikingLIFFuncMultiLayer(SNNFuncMultiLayer):

    def __init__(self, network_s_list, network_x, vth, leaky, fb_num=1):
        super(SNNBERTSpikingLIFFuncMultiLayer, self).__init__(network_s_list, network_x, vth, fb_num)
        self.leaky = torch.tensor(leaky, requires_grad=False)

    def snn_forward(self, x, segment_ids, time_step, output_type='normal', input_type='constant', attention_mask=None):

        if input_type == 'constant':
            x1 = self.network_x(x, segment_ids)
        fac = 1.
        attn_list = []
        u_list = []
        s_list = []
        u1 = x1
        s1 = (u1 >= self.vth).float()

        #s2 = (u1 <= -1 * self.vth).float()
        #s1 = s1 - s2

        u1 = u1 - self.vth * s1
        # add leaky term here
        u1 = u1 * self.leaky

        u_list.append(u1)
        s_list.append(s1)
        for i in range(len(self.network_s_list) - 1):
            if i % 6 == 0:
                # Key Layer
                ui = self.network_s_list[i](s_list[-1])
            elif i % 6 == 1:
                # Value Layer
                ui = self.network_s_list[i](s_list[-2])
            elif i % 6 == 2:
                # Attention Layer
                ui, layer_attn = self.network_s_list[i](s_list[-3], s_list[-2], s_list[-1], attention_mask)
                attn_list.append(layer_attn)
            elif i % 6 == 3:
                # Self Output Layer
                ui = self.network_s_list[i](s_list[-1], s_list[-4])
            elif i % 6 == 5:
                # Output Layer
                ui = self.network_s_list[i](s_list[-1], s_list[-2])
            else:
                ui = self.network_s_list[i](s_list[-1])

            if i%6 in [0,1]:
                si = (ui >= fac*self.vth).float()
                ui = ui - fac*self.vth * si
            else:
                si = (ui >= self.vth).float()
                ui = ui - self.vth * si

            #s2 = (ui <= -1 * self.vth).float()
            #si = si - s2

            #Commented out
            #ui = ui - self.vth * si

            # add leaky term here
            ui = ui * self.leaky

            u_list.append(ui)
            s_list.append(si)

        af = s_list[0]
        al = s_list[-self.fb_num]

        a_per_layer = []
        avg_attn = []
        avg_print = []
        avg_print_later = []
        list_avg = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        l_id = 0
        l_later = 1
        test_conv_first_attn = []
        #print('Spikes : ', s_list[l_id][0][0][-200:-50])
        # List of average values
        layer_dbg = 2
        for i in range(len(s_list)):
            a_per_layer.append(s_list[i])
            list_avg[i].append(torch.mean(a_per_layer[i]).item())
            if i == l_id:
                avg_print.append(torch.mean(a_per_layer[i]).item())
            if i == l_later:
                avg_print_later.append(torch.mean(a_per_layer[i]).item())
            if i == layer_dbg:
                test_conv_first_attn.append(a_per_layer[i][0])
        for i in range(len(attn_list)):
            avg_attn.append(attn_list[i])

        if output_type == 'all_rate':
            r_list = []
            for s in s_list:
                r_list.append(s)

        for t in range(time_step - 1):
            if input_type == 'constant':
                # last layer is output layer which takes two inputs

                # When feedback is used
                #u_list[0] = u_list[0] + self.network_s_list[-1](s_list[-1]) + x1

                # No feed back case
                u_list[0] = u_list[0] + x1
            s_list[0] = (u_list[0] >= self.vth).float()

            #s2 = (u_list[0] <= -1 * self.vth).float()
            #s_list[0] = s_list[0] - s2

            u_list[0] = u_list[0] - self.vth * s_list[0]
            # add leaky term here
            u_list[0] = u_list[0] * self.leaky

            for i in range(len(self.network_s_list) - 1):
                if i % 6 == 0:
                    # Key Layer
                    u_list[i + 1] = u_list[i + 1]  + self.network_s_list[i](s_list[i])
                elif i % 6 == 1:
                    # Value Layer
                    u_list[i + 1] = u_list[i + 1]  + self.network_s_list[i](s_list[i-1])
                elif i % 6 == 2:
                    # Attention Layer
                    u_val, layer_attn = self.network_s_list[i](s_list[i-2], s_list[i-1], s_list[i],  attention_mask)
                    u_list[i + 1] = u_list[i + 1] + u_val
                    attn_list[int(i / 6)] = layer_attn
                elif i % 6 == 3:
                    # Self Output Layer
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i], s_list[i-3])
                elif i % 6 == 5:
                    # Output Layer
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i], s_list[i-1])
                else:
                    u_list[i + 1] = u_list[i + 1] + self.network_s_list[i](s_list[i])

                
                if i%6 in [0,1]:
                    s_list[i + 1] = (u_list[i + 1] >= fac*self.vth).float()
                else:
                    s_list[i + 1] = (u_list[i + 1] >= self.vth).float()

                #s2 = (u_list[i + 1] <= -1 * self.vth).float()
                #s_list[i + 1] = s_list[i + 1] - s2

                if i%6 in [0,1]:
                    u_list[i + 1] = u_list[i + 1] - fac * self.vth * s_list[i + 1]
                else:
                    u_list[i + 1] = u_list[i + 1] - self.vth * s_list[i + 1]
                # add leaky term here
                u_list[i + 1] = u_list[i + 1] * self.leaky
                # print('Lyaer : ', i+1)
                #print('Spikes' , s_list[i+1][0][0][:100])
            af = af * self.leaky + s_list[0]
            al = al * self.leaky + s_list[-self.fb_num]
            #print('Vth ', self.vth)
            for layer_num in range(len(s_list)):
                a_per_layer[layer_num] = a_per_layer[layer_num] + s_list[layer_num] #a_per_layer[layer_num] * self.leaky + s_list[layer_num]
                list_avg[layer_num].append(torch.mean(a_per_layer[layer_num][0]/(t+2)).item())
                if layer_num == l_id:
                    avg_print.append((torch.mean(a_per_layer[l_id])/(t+2)).item())
                if layer_num == l_later:
                    avg_print_later.append((torch.mean(a_per_layer[l_later])/(t+2)).item())
                if layer_num == layer_dbg and t % 5 == 0:
                    test_conv_first_attn.append(a_per_layer[layer_num][0]/(t+2))

            for layer_num in range(len(attn_list)):
                avg_attn[layer_num] = avg_attn[layer_num] + attn_list[layer_num]

            if output_type == 'all_rate':
                for i in range(len(r_list)):
                    r_list[i] = r_list[i] + s_list[i]


        # Uncomment to see ASR behavior
        iu = 0
        count_layer = 0
        sum_layer = 0
        # if random.uniform(0,1) < 0.01:
        #     for avg in list_avg:
        #         precision = 3
        #         iu +=1
        #         avg_val = sum(avg)/len(avg)
        #         count_layer += 1
        #         sum_layer += avg_val
        #         formatted_list = [f"{num:.{precision}f}" for num in avg]
        #         print('Layer ', iu)
        #         print('Avg spiking rate : ', formatted_list) #[-5:])
        #         print('Avg Value across time :', avg_val)
        #     print('Net avg. spikes: ', sum_layer/count_layer)

        weighted = ((1. - self.leaky ** time_step) / (1. - self.leaky))
        if output_type == 'normal':
            return af / weighted, al / weighted
        elif output_type == 'all_layers':
            for layer_num in range(len(s_list)):
                a_per_layer[layer_num] = a_per_layer[layer_num]  * (1.0 / time_step) #/ weighted
            #for layer_num in range(len(avg_attn)):
            #    avg_attn[layer_num] = avg_attn[layer_num]  * (1.0 / time_step) #/ weighted
            return a_per_layer, avg_attn, test_conv_first_attn
        elif output_type == 'all_rate':
            for i in range(len(r_list)):
                r_list[i] *= 1.0 / time_step
            return r_list
        elif output_type == 'first':
            return af / weighted
        else:
            return al / weighted


class SNNFC(nn.Module):

    def __init__(self, d_in, d_out, bias=False, need_resize=False, sizes=None, dropout=0.0):
        super(SNNFC, self).__init__()
        self.fc = nn.Linear(d_in, d_out, bias=bias)
        self.need_resize = need_resize
        self.sizes=sizes
        self.drop = nn.Dropout(dropout)

        self._initialize_weights()

    def forward(self, x):
        if self.need_resize:
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        return self.drop(x)

    def forward_linear(self, x):
        if self.need_resize:
            if self.sizes == None:
                sizes = x.size()
                B = sizes[0]
                x = torch.reshape(self.fc(x.reshape(B, -1)), sizes)
            else:
                B = x.size(0)
                self.sizes[0] = B
                x = torch.reshape(self.fc(x.reshape(B, -1)), self.sizes)
        else:
            x = self.fc(x)
        return x

    def _wnorm(self, norm_range=1.):
        self.fc, self.fc_fn = weight_spectral_norm(self.fc, names=['weight'], dim=0, norm_range=norm_range)

    def _reset(self, x):
        if 'fc_fn' in self.__dict__:
            self.fc_fn.reset(self.fc)
        self.drop.reset_mask(x)

    def _initialize_weights(self):
        m = self.fc
        m.weight.data.uniform_(-1, 1)
        for i in range(m.weight.size(0)):
            m.weight.data[i] /= torch.norm(m.weight.data[i])
        if m.bias is not None:
            m.bias.data.zero_()

    def copy(self, target):
        self.fc.weight.data = target.fc.weight.data.clone()
        if self.fc.bias is not None:
            self.fc.bias.data = target.fc.bias.data.clone()
        self.need_resize = target.need_resize
        self.sizes = target.sizes


