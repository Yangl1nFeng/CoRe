from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange

from model.vision.basic_modules import _get_activation_fn, _get_clones

class Attention(nn.Module):
    def __init__(self, region_num, feat_dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(feat_dim, feat_dim * 3, bias=bias)
        
        self.project_out = nn.Linear(feat_dim, feat_dim, bias=bias)
        
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.attn5 = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        self.attn6 = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        # self.attn7 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.attn8 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x, tgt_mask=None):
        b, n, c = x.shape  # b: batch_size, n: region_num, c: feat_dim

        # 将输入通过线性层转换为 q, k, v
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 调整为多头注意力的形状
        q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(k, 'b n (h c) -> b h n c', h=self.num_heads)
        v = rearrange(v, 'b n (h c) -> b h n c', h=self.num_heads)

        # 对 q 和 k 进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力分数（缩放点积注意力）
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        if tgt_mask is not None:
            tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, region_num, 1]
            tgt_mask = tgt_mask.expand(-1, self.num_heads, -1, n)  # [batch_size, num_heads, region_num, region_num]
            attn = attn.masked_fill(tgt_mask == 0, float('-inf'))

        mask1 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)

        mask5 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)
        mask6 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)
        mask7 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)
        mask8 = torch.zeros(b, self.num_heads, n, n, device=x.device, requires_grad=False)

        index = torch.topk(attn, k=int(n*1/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n*2/2), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n*3/8), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n*4/8), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n*5/8), dim=-1, largest=True)[1]
        mask5.scatter_(-1, index, 1.)
        attn5 = torch.where(mask5 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n*6/8), dim=-1, largest=True)[1]
        mask6.scatter_(-1, index, 1.)
        attn6 = torch.where(mask6 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n*7/8), dim=-1, largest=True)[1]
        mask7.scatter_(-1, index, 1.)
        attn7 = torch.where(mask7 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(n), dim=-1, largest=True)[1]
        mask8.scatter_(-1, index, 1.)
        attn8 = torch.where(mask8 > 0, attn, torch.full_like(attn, float('-inf')))

        # 对每个注意力矩阵应用 softmax
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        attn5 = attn5.softmax(dim=-1)
        attn6 = attn6.softmax(dim=-1)
        # attn7 = attn3.softmax(dim=-1)
        # attn8 = attn4.softmax(dim=-1)

        out1 = attn1 @ v
        out2 = attn2 @ v
        out3 = attn3 @ v
        out4 = attn4 @ v

        out5 = attn5 @ v
        out6 = attn6 @ v
        # out7 = attn7 @ v
        # out8 = attn8 @ v

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4 +   out5 * self.attn5 + out6 * self.attn6 # + out7 * self.attn7 + out8 * self.attn8

        out = rearrange(out, 'b h n c -> b n (h c)')
        out = self.project_out(out)
        
        return out

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()


        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self, tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
    
        self.self_attn = Attention(230,d_model,nhead,bias=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self, tgt, tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = tgt
        tgt2 = self.self_attn(tgt2, tgt_mask)
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, None

    
class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
        self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
        spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' %(d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))

    def forward(self, q, k, v, pairwise_locs, key_padding_mask=None):
        residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t') 
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual)
            spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head, d=self.spatial_dim+1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)

        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)
        
        assert torch.sum(torch.isnan(fused_attn) == 0), print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, fused_attn
    
class TransformerSpatialDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
        spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout, 
            spatial_multihead=spatial_multihead, 
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
        self, tgt, memory, tgt_pairwise_locs,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):

        tgt2 = self.norm1(tgt)
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, cross_attn_matrices = self.multihead_attn(
            query=tgt2, key=memory,
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, self_attn_matrices, cross_attn_matrices

class TransformerSpatialEncoderLayer(TransformerEncoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
        spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul'
    ):
        super().__init__(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation
        )
        del self.self_attn
        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout, 
            spatial_multihead=spatial_multihead, 
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

    def forward(
        self, tgt, tgt_pairwise_locs,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
    ):

        tgt2 = tgt
        tgt2, self_attn_matrices = self.self_attn(
            tgt2, tgt2, tgt2, tgt_pairwise_locs,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, self_attn_matrices
