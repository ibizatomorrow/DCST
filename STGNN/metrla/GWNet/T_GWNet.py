import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import argparse

from STGNN.metrla.GWNet import util
import yaml


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gwnet(nn.Module):
    def __init__(self):
        super(gwnet, self).__init__()

        self.get_args()

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.adjinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device), requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device), requires_grad=True).to(self.device)
                self.supports_len +=1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.adjinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1




        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1,self.kernel_size),dilation=new_dilation)) 

                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(self.dilation_channels,self.residual_channels,self.dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                  out_channels=self.end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    
    def get_args(self):
        file_path = "../STGNN/metrla/GWNet/gwnet.yaml"
        with open(file_path, "r") as f:
            cfg_gwnet = yaml.safe_load(f)

        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(cfg_gwnet.get("adjdata"), cfg_gwnet.get("adjtype"))
        self.device = torch.device(cfg_gwnet.get("device"))
        self.supports = [torch.tensor(i).to(self.device) for i in adj_mx]

        if cfg_gwnet.get("randomadj"):
            self.adjinit = None
        else:
            self.adjinit = self.supports[0]

        if cfg_gwnet.get("aptonly"):
            self.supports = None

        self.num_nodes = cfg_gwnet.get("num_nodes")
        self.dropout = cfg_gwnet.get("dropout")
        self.gcn_bool = cfg_gwnet.get("gcn_bool")
        self.addaptadj = cfg_gwnet.get("addaptadj")
        self.in_dim = cfg_gwnet.get("in_dim")
        self.out_dim = cfg_gwnet.get("out_dim")
        self.residual_channels = cfg_gwnet.get("nhid")
        self.dilation_channels = cfg_gwnet.get("nhid")
        self.skip_channels = cfg_gwnet.get("nhid")*8
        self.end_channels = cfg_gwnet.get("nhid")*16
        self.kernel_size = cfg_gwnet.get("kernel_size")
        self.blocks = cfg_gwnet.get("blocks")
        self.layers = cfg_gwnet.get("layers")

        
