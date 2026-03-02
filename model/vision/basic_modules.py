import copy

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np
class SharedConv1d(nn.Module):
    def __init__(self, kernel_size):
        super(SharedConv1d, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size) 
    
    def forward(self, x):
        batch_size, in_channels, length = x.shape
        x = x.reshape(batch_size * in_channels, 1, length)
        x = self.conv(x)
        _, _, new_length = x.shape
        x = x.reshape(batch_size, in_channels, new_length)
        
        return x
    

class RoMa(nn.Module):
    def __init__(self, d_hidden):
        super(RoMa, self).__init__()
        self.linear1 = SharedConv1d(kernel_size=d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_hidden)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """

        max_len = features.size(1)
        mask = torch.arange(max_len).expand(features.size(0), features.size(1)).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        mask_features = features.masked_fill(mask == 0, 0)

        features_external = self.linear1(mask_features)
        features_external = torch.div(features_external,768)
        features_external_ = self.linear2(features)

        features_k_softmax1 = nn.Softmax(dim=1)(features_external)
        features_k_softmax1_ = nn.Softmax(dim=1)(features_external_)
        
        pool_features = torch.sum(1.5*features_k_softmax1 * 0.67*features_k_softmax1_ * features,dim=1)

        return pool_features
    
class AP(nn.Module):
    def __init__(self, d_hidden):
        super(AP, self).__init__()
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.linear_sorted = nn.Linear(d_hidden, 1)
        self.balance = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.linear.in_features +
                                  self.linear.out_features)
        self.linear.weight.data.uniform_(-r, r)
        self.linear.bias.data.fill_(0)

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """

        max_len = features.size(1)
        mask = torch.arange(max_len).expand(features.size(0), features.size(1)).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        mask_features = features.masked_fill(mask == 0, 0)
        # mask_features = mask_features.sort(dim=1, descending=True)[0]

        feat_transformed = self.linear(mask_features) # self.dropout(self.linear(mask_features))
        attention_scores = feat_transformed.masked_fill(mask == 0, -10000)  # Apply mask
        attention_weights = nn.Softmax(dim=1)(attention_scores-torch.max(attention_scores,dim=1)[0].unsqueeze(1))
        attn = attention_weights.masked_fill(mask == 0, 0)  # Apply mask
        token_features = torch.sum(features * attn, dim=1)  # Shape: [batch_size, embed_size]

        weight = self.relu(self.linear_sorted(feat_transformed))  # 【batch_size, obj】
        weight = nn.Softmax(dim=1)(weight-torch.max(weight,dim=1)[0].unsqueeze(1))
        embed_features = torch.sum(weight * features, dim=1)  # [batch_size, obj, feat_dim]
        fusion_features = torch.cat([token_features.unsqueeze(1),
                                     embed_features.unsqueeze(1)],
                                    dim=1)
        fusion_weights = F.softmax(self.balance(fusion_features),
                                   dim=1)
        pool_features = (fusion_features * fusion_weights).sum(1)

        return pool_features

class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        """
        Sparsemax activation function.

        Args:
            dim (int): Dimension along which to apply sparsemax. Default is -1.
        """
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        """
        Forward pass for Sparsemax.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Sparsemax output tensor.
        """
        sorted_input, _ = torch.sort(input, descending=True, dim=self.dim)
        cumsum_sorted = torch.cumsum(sorted_input, dim=self.dim)
        range_tensor = torch.arange(1, input.size(self.dim) + 1, device=input.device).view(
            [1 if i != self.dim else -1 for i in range(input.dim())]
        )
        condition = sorted_input - (cumsum_sorted - 1) / range_tensor > 0
        k = condition.sum(dim=self.dim, keepdim=True)
        tau = (torch.gather(cumsum_sorted, self.dim, k - 1) - 1) / k
        output = torch.clamp(input - tau, min=0)
        
        return output
    
def get_mlp_head(input_size, hidden_size, output_size, dropout=0):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size//2),
        nn.ReLU(),
        nn.LayerNorm(hidden_size//2, eps=1e-12),
        nn.Dropout(dropout),
        nn.Linear(hidden_size//2, output_size)
        )
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def calc_pairwise_locs(obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center', spatial_dist_norm=True, spatial_dim=5):
    if pairwise_rel_type == 'mlp':
        obj_locs = torch.cat([obj_centers, obj_whls], 2)
        pairwise_locs = torch.cat(
            [einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
            einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))],
            dim=3
        )
        return pairwise_locs

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps) # (b, l, l)
    if spatial_dist_norm:
        max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
        norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
    else:
        norm_pairwise_dists = pairwise_dists

    if spatial_dim == 1:
        return norm_pairwise_dists.unsqueeze(3)
        
    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2]**2, 3)+eps)
    if pairwise_rel_type == 'center':
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, pairwise_locs[..., 2]/pairwise_dists, 
            pairwise_dists_2d/pairwise_dists, pairwise_locs[..., 1]/pairwise_dists_2d,
            pairwise_locs[..., 0]/pairwise_dists_2d],
            dim=3
        )
    elif pairwise_rel_type == 'vertical_bottom':
        bottom_centers = torch.clone(obj_centers)
        bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
        bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
            - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
        bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs**2, 3) + eps) # (b, l, l)
        bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2]**2, 3)+eps)
        pairwise_locs = torch.stack(
            [norm_pairwise_dists, 
            bottom_pairwise_locs[..., 2]/bottom_pairwise_dists, 
            bottom_pairwise_dists_2d/bottom_pairwise_dists, 
            pairwise_locs[..., 1]/pairwise_dists_2d,
            pairwise_locs[..., 0]/pairwise_dists_2d],
            dim=3
        )
        
    if spatial_dim == 4:
        pairwise_locs = pairwise_locs[..., 1:]
    return pairwise_locs

class ViewWeightMLP(nn.Module):
    def __init__(self, input_dim):
        super(ViewWeightMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)  
        ).cuda()
    
    def forward(self, x):
        scores = self.mlp(x) 
        scores = scores.squeeze(-1) 
        scores = scores / 1
        weights = F.softmax(scores, dim=1) 
        return weights
    
class MappingMLP(nn.Module):
    def __init__(self, input_dim):
        super(MappingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024), 
            nn.ReLU(),
            nn.Linear(1024, input_dim)  
        ).cuda()
    
    def forward(self, x):
        feature = self.mlp(x) 
        return feature
    
def get_mixup_function(mixup_strategy, mixup_stage1, mixup_stage2):
    if mixup_strategy is None:
        return None
    assert mixup_strategy in ['linear_decay', 'all_mixup']
    
    if mixup_strategy == 'linear_decay':
        return LinearDecayMixup(mixup_stage1, mixup_stage2)
    elif mixup_strategy == 'all_mixup':
        return AllMixup()

class AllMixup:
    def __init__(self) -> None:
        pass
    
    def __call__(self, obj_sem_cls_pred, obj_labels, cur_step, total_steps):
        mixup_sem_cls_pred = torch.zeros_like(obj_sem_cls_pred)
        for i in range(mixup_sem_cls_pred.shape[0]):
            for j in range(mixup_sem_cls_pred.shape[1]):
                if obj_labels[i, j] >= 0:
                    mixup_sem_cls_pred[i, j, obj_labels[i, j]] = 1.0
        return mixup_sem_cls_pred
        
class LinearDecayMixup:
    def __init__(self, mixup_stage1, mixup_stage2) -> None:
        self.stage1_rate = mixup_stage1
        self.stage2_rate = mixup_stage2
        assert self.stage2_rate > self.stage1_rate
    
    def __call__(self, obj_sem_cls_pred, obj_labels, cur_step, total_steps):
        if cur_step < total_steps * self.stage1_rate:
            mixup_ratio = 1.0
        elif cur_step < total_steps * self.stage2_rate:
            mixup_ratio = (total_steps * self.stage2_rate - cur_step) / ((self.stage2_rate - self.stage1_rate) * total_steps)
        else:
            mixup_ratio = 0.0
        # mixup
        mixup_sem_cls_pred = obj_sem_cls_pred.clone() # B, O, 607
        random_numer = torch.rand(mixup_sem_cls_pred.shape[0:2]) # B, O
        mixup_mask = random_numer < mixup_ratio
        for i in range(mixup_sem_cls_pred.shape[0]):
            for j in range(mixup_sem_cls_pred.shape[1]):
                if mixup_mask[i, j] and obj_labels[i, j] >= 0:
                    mixup_sem_cls_pred[i, j, :] = 0.0
                    mixup_sem_cls_pred[i, j, obj_labels[i, j]] = 1.0
        return mixup_sem_cls_pred

class Attn_Fusion(nn.Module):
    def __init__(self, embed_size):
        super(Attn_Fusion, self).__init__()
        self.linear = nn.Linear(embed_size, embed_size).cuda()
        self.linear1 = nn.Linear(embed_size, embed_size).cuda()
    
    def forward(self, feat, feat_len):
        """
        feat: Tensor of shape [batch_size, region_num, embed_size]
        feat_len: Tensor of shape [batch_size, region_num] with values 0 (False) or 1 (True)
        """
        
        feat = self.linear(feat) 
        feat_transformed = self.linear1(feat)
        mask = feat_len.unsqueeze(-1)  # Shape: [batch_size, region_num]
        attention_scores = feat_transformed.masked_fill(mask == 0, -float('inf'))  # Apply mask
        
        attention_weights = nn.Softmax(dim=1)(attention_scores-torch.max(attention_scores,dim=1)[0].unsqueeze(1))
        attn = attention_weights.masked_fill(mask == 0, 0)  # Apply mask
        feat_weighted = torch.sum(feat * attn, dim=1)  # Shape: [batch_size, embed_size]
        return feat_weighted
        

class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True).cuda()
        self.linear = nn.Linear(self.d_hidden, 1, bias=False).cuda()

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.03, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe     

def generate_causal_mask(length):
    return (torch.triu(torch.ones(length, length))==1).transpose(0,1)

def generate_mm_casual_mask(txt_length, pc_length):
    txt_mask = torch.cat((generate_causal_mask(txt_length), torch.ones(txt_length, pc_length)==1), dim=1)
    pc_mask = torch.cat((torch.zeros(pc_length, txt_length), generate_causal_mask(pc_length)), dim=1)
    return torch.cat((txt_mask, pc_mask), dim=0)