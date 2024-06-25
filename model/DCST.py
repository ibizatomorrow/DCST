import torch.nn as nn
import torch
from torchinfo import summary
import numpy as np
import pandas as pd
import os
from einops import rearrange, repeat
import math

class ViewMerging(nn.Module):
    def __init__(self, win_size, model_dim):
        super(ViewMerging, self).__init__()

        self.win_size = win_size
        self.model_dim = model_dim

        self.temporal_merge = nn.Linear(win_size*model_dim, model_dim)
        self.norm = nn.LayerNorm(win_size*model_dim)


    def forward(self, x):
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)
        x = self.norm(x)
        x = self.temporal_merge(x)
        return x

class Temporal_scale(nn.Module):
    def __init__(
        self, win_size, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        #Merge Temporal Points to Temporal Segment
        self.merge_layer = ViewMerging(win_size, model_dim)

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)

        #Temporal Segment
        x_seg = self.merge_layer(x)

        residual = x
        out = self.attn(x, x_seg, x_seg)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
    
class TemporalATT(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        #Length of Temporal Segment
        self.temporal_size = [1, 2, 3, 4]

        self.temporal_blocks = nn.ModuleList()

        for i in range(len(self.temporal_size)):
            self.temporal_blocks.append(Temporal_scale(self.temporal_size[i], model_dim, feed_forward_dim, num_heads, dropout, mask))

    def forward(self, x, dim=-2):
        for block in self.temporal_blocks:
            x = block(x, dim)

        return x
    
class node2grid_encoder(nn.Module):
    def __init__(self, view, d_model):
        super(node2grid_encoder, self).__init__()
        self.view = view
        self.d_model = d_model
        self.device = self.view[0].device

        #One linear layer per node
        self.N2Gencoder_w = torch.randn(self.view.shape[1], self.d_model, self.d_model).to(self.device)
        self.N2Gencoder_w = nn.Parameter(self.N2Gencoder_w)
        self.N2Gencoder_b = torch.randn(1, self.view.shape[1], self.d_model).to(self.device)
        self.N2Gencoder_b = nn.Parameter(self.N2Gencoder_b)
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        batch = x.shape[0]
        x = torch.einsum("btni,nio->btno", x, self.N2Gencoder_w)
        x = rearrange(x, 'b t_num node_num d -> (b t_num) node_num d', t_num = x.shape[1])
        x = x + self.N2Gencoder_b
        x = rearrange(x, '(b t_num) node_num d-> b t_num node_num d', b = batch)
        x_grid_embed = torch.einsum("btnd,gn->btgd", x, self.view)

        x_grid_embed = self.norm(x_grid_embed)
        return x_grid_embed

class Spatial_scale(nn.Module):
    def __init__(
        self, view, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        #Spatial Node to Spatial Grid
        if(view.size(0) == 1):
            self.node2grid = None
        else:
            self.node2grid= node2grid_encoder(view, model_dim)

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)

        if self.node2grid is not None:
            x_grid = self.node2grid(x)
        else:
            x_grid = x

        residual = x
        out = self.attn(x, x_grid, x_grid)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class Spatial_ATT(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        # Size of Spatial Grid
        self.view_size = [160, 80, 40]

        self.device = torch.device('cuda:0')

        #get grid_node view information
        self.get_view_info(self.view_size)

        self.spatial_blocks = nn.ModuleList()
        self.spatial_blocks.append(Spatial_scale(torch.tensor([1]), model_dim, feed_forward_dim, num_heads, dropout, mask))

        for i in range(len(self.views)):
            self.spatial_blocks.append(Spatial_scale(self.views[i], model_dim, feed_forward_dim, num_heads, dropout, mask))
        
    def forward(self, x, dim=-2):
        for block in self.spatial_blocks:
            x = block(x, dim)
        
        return x



    def get_view_info(self, view_size):
    #Cross-scale
        views = []
        for i in range(len(view_size)):
            grid_node_path = "grid_node_" + str(view_size[i]) + ".csv"
            grid_node = pd.read_csv(os.path.join("../data",
                                "METRLA", grid_node_path))
            grid_node = grid_node.values
            grid_node = grid_node[:, 1:]

            grid_sum = np.sum(grid_node, axis=1)
            grid_mean = np.reciprocal(grid_sum)
            grid_mean = grid_mean.reshape(len(grid_mean),1)
            grid_mean_repeat = grid_mean.repeat(repeats=np.size(grid_node, 1), axis = 1)
            grid_node = np.multiply(grid_node, grid_mean_repeat)

            grid_node = torch.from_numpy(grid_node).float().to(self.device)
            views.append(grid_node)
        self.views = views



class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class DCST(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        #DCST
        self.attn_layers_t_revised = TemporalATT(self.model_dim, feed_forward_dim, num_heads, dropout)
        self.attn_layers_s_revised = Spatial_ATT(self.model_dim, feed_forward_dim, num_heads, dropout)



    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        #DCST
        x = self.attn_layers_t_revised(x, dim=1)
        x = self.attn_layers_s_revised(x, dim=2)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out
