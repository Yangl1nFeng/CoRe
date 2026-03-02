import json
import os

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataset.path_config import SCAN_FAMILY_BASE
from model.vision.basic_modules import (_get_clones, calc_pairwise_locs,
                                        get_mlp_head, init_weights, get_mixup_function)
from model.vision.pointnet2.pointnet2_modules import PointnetSAModule
from model.vision.transformers import (TransformerDecoderLayer,
                                       TransformerEncoderLayer,
                                       TransformerSpatialDecoderLayer,
                                       TransformerSpatialEncoderLayer)
from pipeline.registry import registry

# 创建随机特征mask（10% mask掉）
def generate_feature_mask(obj_masks, feat_dim, mask_prob=0.1):
    batch_size, object_num = obj_masks.shape
    
    # 随机生成一个 [batch_size, object_num, feat_dim] 的矩阵，值在 [0, 1) 之间
    rand_vals = torch.rand(batch_size, object_num, feat_dim)
    
    # 对于有效物体，选择 mask_prob 的概率进行 masking
    feature_masks = rand_vals > mask_prob
    
    # 将无效物体的特征全部置为 1（不做mask）
    feature_masks = feature_masks.cuda() * obj_masks.unsqueeze(-1)
    
    return feature_masks

def generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10.0):
    """
    生成空间特征的 mask，其中目标物体和更近的物体更有可能被 mask。
    
    掩码概率规则：
        - 距离目标物体小于某个阈值时，掩码概率为 15%。
        - 距离目标物体较远时，掩码概率线性递减，最大为 15%。

    参数:
        target_id: [batch_size, 1]，目标物体的索引。
        obj_masks: [batch_size, 80]，原有的物体有效性掩码。
        pairwise_locs: [batch_size, 4, 80, 80, 5]，多视角物体的两两空间关系。
        mask_prob: 最大的掩码概率，默认是 15%。
        max_dist: 用于计算物体之间的最大距离，以规范化距离。

    返回:
        mask: [batch_size, 80]，生成的掩码，表示哪些物体被 mask。
    """
    batch_size, num_objects = obj_masks.size()

    # 提取第一个视角的物体间的距离
    pairwise_dists = pairwise_locs[:, 0, :, :, 0]  # [batch_size, 80, 80]

    # 初始化掩码概率矩阵
    mask_probs = torch.full_like(pairwise_dists, mask_prob)  # [batch_size, 80, 80]

    # 获取目标物体的索引
    target_idx = target_id.squeeze(1)  # [batch_size]

    # 计算目标物体与所有其他物体的距离
    # Gather target distance (this should be [batch_size, 80, 1])
    dists_to_target = pairwise_dists.gather(2, target_idx.unsqueeze(1).unsqueeze(2).expand(-1, num_objects, -1))  # [batch_size, 80, 1]
    
    # 扩展 dists_to_target 为 [batch_size, 80, 80]，这样可以与 mask_probs 进行元素级比较
    dists_to_target = dists_to_target.expand(-1, num_objects, num_objects)  # [batch_size, 80, 80]

    # 基于距离调整掩码概率
    # 距离小于 max_dist 时，掩码概率为 15%
    mask_probs[dists_to_target < max_dist] = mask_prob

    # 对于距离大于 max_dist 的物体，线性递减掩码概率
    # 掩码概率随着距离的增大线性递减，最终趋向于 0
    mask_probs[dists_to_target >= max_dist] = mask_prob * (1.0 - (dists_to_target[dists_to_target >= max_dist] - max_dist) / max_dist)
    mask_probs = torch.clamp(mask_probs, min=0.0, max=mask_prob)  # 确保掩码概率在 [0, mask_prob] 之间

    # 将目标物体本身的掩码概率设为 mask_prob，确保目标物体有 15% 的掩码概率
    mask_probs.scatter_(2, target_idx.unsqueeze(1).unsqueeze(2), mask_prob)  # [batch_size, 80, 80]

    # 使用概率生成最终的 mask
    rand_vals = torch.rand_like(mask_probs).to(torch.float32)  # 生成随机值 [batch_size, 80, 80]
    mask = (rand_vals < mask_probs).float()  # 根据概率生成掩码

    # 保证物体掩码与有效物体相匹配
    mask = mask * obj_masks.unsqueeze(2)  # [batch_size, 80, 80] * [batch_size, 80, 1]

    # 将生成的掩码压缩到每个物体上，返回每个物体的掩码
    mask = mask.max(dim=2)[0]  # [batch_size, 80]

    return mask




def break_up_pc(pc: Tensor):
    """
    Split the pointcloud into xyz positions and features tensors.
    This method is taken from VoteNet codebase (https://github.com/facebookresearch/votenet)

    @param pc: pointcloud [N, 3 + C]
    :return: the xyz tensor and the feature tensor
    """
    xyz = pc[..., 0:3].contiguous()
    features = (
        pc[..., 3:].transpose(1, 2).contiguous()
        if pc.size(-1) > 3 else None
    )
    return xyz, features

class PointNetPP(nn.Module):
    """
    Pointnet++ encoder.
    For the hyper parameters please advise the paper (https://arxiv.org/abs/1706.02413)
    """

    def __init__(self, sa_n_points: list,
                 sa_n_samples: list,
                 sa_radii: list,
                 sa_mlps: list,
                 bn=True,
                 use_xyz=True):
        super().__init__()

        n_sa = len(sa_n_points)
        if not (n_sa == len(sa_n_samples) == len(sa_radii) == len(sa_mlps)):
            raise ValueError('Lens of given hyper-params are not compatible')

        self.encoder = nn.ModuleList()

        for i in range(n_sa):
            self.encoder.append(PointnetSAModule(
                npoint=sa_n_points[i],
                nsample=sa_n_samples[i],
                radius=sa_radii[i],
                mlp=sa_mlps[i],
                bn=bn,
                use_xyz=use_xyz,
            ))

        out_n_points = sa_n_points[-1] if sa_n_points[-1] is not None else 1
        self.fc = nn.Linear(out_n_points * sa_mlps[-1][-1], sa_mlps[-1][-1])

    def forward(self, features):
        """
        @param features: B x N_objects x N_Points x 3 + C
        """
        xyz, features = break_up_pc(features)
        for i in range(len(self.encoder)):
            xyz, features = self.encoder[i](xyz, features)

        return self.fc(features.view(features.size(0), -1))

@registry.register_vision_model("pointnet_point_encoder")
class PcdObjEncoder(nn.Module):
    def __init__(self, path=None, freeze=False):
        super().__init__()

        self.pcd_net = PointNetPP(
            sa_n_points=[32, 16, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
        )
        
        self.obj3d_clf_pre_head = get_mlp_head(768, 768, 607, dropout=0.3)
        
        self.dropout = nn.Dropout(0.1)
        
        if path is not None:
            state_dict = torch.load(path)
            d1= [(key.removeprefix("obj_encoder."), val) for key, val in state_dict.items() if key.startswith("obj_encoder")]
            d2 = [(key, val) for key, val in state_dict.items() if key.startswith("obj3d_clf_pre_head")]
            self.load_state_dict(dict(d1 + d2))
        
        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
    
    def freeze_bn(self, m):
        '''Freeze BatchNorm Layers'''
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                
    def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks):
        if self.freeze:
            self.freeze_bn(self.pcd_net)
            
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.pcd_net(einops.rearrange(obj_pcds, 'b o p d -> (b o) p d') )
        obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        obj_embeds = self.dropout(obj_embeds)
        # freeze
        if self.freeze:
            obj_embeds = obj_embeds.detach()
        # sem logits
        obj_sem_cls = self.obj3d_clf_pre_head(obj_embeds)
        return obj_embeds, obj_embeds, obj_sem_cls

@registry.register_vision_model("point_tokenize_encoder")
class PointTokenizeEncoder(nn.Module):
    def __init__(self, backbone='pointnet++', hidden_size=768, path=None, freeze_feature=False,
                num_attention_heads=12, spatial_dim=5, num_layers=4, dim_loc=6, pairwise_rel_type='center',
                mixup_strategy=None, mixup_stage1=None, mixup_stage2=None):
        super().__init__()
        #  print(spatial_dim, dim_loc, num_attention_heads)
        assert backbone in ['pointnet++', 'pointnext']
        
        # build backbone
        if backbone == 'pointnet++':
            self.point_feature_extractor = PointNetPP(
                sa_n_points=[32, 16, None],
                sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None],
                sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
            )
        elif backbone == 'pointnext':
            self.point_feature_extractor = PointNext()
                      
        # build cls head
        self.point_cls_head = get_mlp_head(hidden_size, hidden_size, 607, dropout=0.0)
        self.dropout = nn.Dropout(0.1) 
        
        # freeze feature
        self.freeze_feature = freeze_feature
        if freeze_feature:
            for p in self.parameters():
                p.requires_grad = False
        
        # build semantic cls embeds
        self.sem_cls_embed_layer = nn.Sequential(nn.Linear(300, hidden_size),
                                                  nn.LayerNorm(hidden_size),
                                                  nn.Dropout(0.1))
        self.int2cat = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.cat2vec = json.load(open(os.path.join(SCAN_FAMILY_BASE, "annotations/meta_data/cat2glove42b.json"), 'r'))  
        # build mask embedes
        self.sem_mask_embeddings = nn.Embedding(1, 768)
        # build spatial encoder layer
        pc_encoder_layer = TransformerSpatialEncoderLayer(hidden_size, num_attention_heads, dim_feedforward=2048, dropout=0.1, activation='gelu', 
                                                       spatial_dim=spatial_dim, spatial_multihead=True, spatial_attn_fusion='cond')
        
        self.spatial_encoder = _get_clones(pc_encoder_layer, num_layers)
        loc_layer = nn.Sequential(
            nn.Linear(dim_loc, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.loc_layers = _get_clones(loc_layer, 1)
        self.pairwise_rel_type = pairwise_rel_type
        self.spatial_dim = spatial_dim
        # build mixup strategy
        self.mixup_strategy = mixup_strategy
        self.mixup_function = get_mixup_function(mixup_strategy, mixup_stage1, mixup_stage2)
        # load weights
        self.apply(init_weights)
        if path is not None:
            self.load_state_dict(torch.load(path), strict=False)
            print('finish load backbone')
        
    
    def freeze_bn(self, m):
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    # def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks, obj_labels=None, cur_step=None, max_steps=None, target_id=None, is_train=False): 
    #     if self.freeze_feature:
    #         self.freeze_bn(self.point_feature_extractor)

    #     # 获取输入尺寸
    #     batch_size, num_objs, view_num, num_points, point_dim = obj_pcds.size()

    #     # 合并 batch 和视角维度进行并行特征提取
    #     obj_pcds_flat = einops.rearrange(obj_pcds, 'b o v p d -> (b o v) p d')
    #     obj_embeds_flat = self.point_feature_extractor(obj_pcds_flat)

    #     # 将特征重新整理为 [batch_size, num_objs, view_num, embed_dim]
    #     obj_embeds = einops.rearrange(obj_embeds_flat, '(b o v) d -> b o v d', b=batch_size, o=num_objs)

    #     # 添加 dropout，并根据设置决定是否冻结特征
    #     obj_embeds = self.dropout(obj_embeds)
    #     if self.freeze_feature:
    #         obj_embeds = obj_embeds.detach()

    #     # 获取语义类别嵌入（视角独立操作，重复广播到视角维度）
    #     obj_sem_cls = self.point_cls_head(obj_embeds)  # B, O, V, 607
    #     if self.freeze_feature:
    #         obj_sem_cls = obj_sem_cls.detach()

    #     if self.mixup_strategy is not None:
    #         obj_sem_cls_mix = self.mixup_function(obj_sem_cls, obj_labels, cur_step, max_steps)
    #     else:
    #         obj_sem_cls_mix = obj_sem_cls.clone()
    #     # 将预测的类别转化为嵌入
    #     obj_sem_cls_mix = torch.argmax(obj_sem_cls_mix, dim=3)  # 按类别维度取最大值
    #     obj_sem_cls_embeds = torch.Tensor([
    #         self.cat2vec[self.int2cat[int(i)]] for i in obj_sem_cls_mix.view(batch_size * num_objs * view_num)
    #     ])
    #     obj_sem_cls_embeds = obj_sem_cls_embeds.view(batch_size, num_objs, view_num, 300).cuda()
    #     obj_sem_cls_embeds = self.sem_cls_embed_layer(obj_sem_cls_embeds)

    #     # 空间关系计算
    #     pairwise_locs = calc_pairwise_locs(
    #         obj_locs[:, :, :, :3], 
    #         obj_locs[:, :, :, 3:], 
    #         pairwise_rel_type=self.pairwise_rel_type,
    #         spatial_dist_norm=True,
    #         spatial_dim=self.spatial_dim
    #     ).squeeze()

    #     # 如果是训练模式，应用 mask
    #     if is_train:
    #         # print(obj_masks.shape)
    #         # obj_embeds_mask = generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10)
    #         # obj_sem_cls_embeds_mask = generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10)
    #         # query_pos_mask = generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10)
    #         # print(obj_embeds_mask)
    #         # obj_embeds = obj_embeds * obj_embeds_mask + obj_sem_cls_embeds * obj_sem_cls_embeds_mask

    #         # 随机mask
            
    #         obj_embeds_mask = generate_feature_mask(obj_masks, obj_embeds.shape[-1], mask_prob=0.1).unsqueeze(2)  # 扩展到视角维度
    #         obj_sem_cls_embeds_mask = generate_feature_mask(obj_masks, obj_embeds.shape[-1], mask_prob=0.15).unsqueeze(2)
    #         query_pos_mask = generate_feature_mask(obj_masks, obj_embeds.shape[-1], mask_prob=0.15).unsqueeze(2)
    #         # print(obj_embeds_mask.shape)
    #         # print(query_pos_mask.shape)
    #         obj_embeds = obj_embeds * obj_embeds_mask + obj_sem_cls_embeds * obj_sem_cls_embeds_mask
    #     else:
    #         obj_embeds = obj_embeds + obj_sem_cls_embeds

    #     # 应用语义掩码
    #     obj_embeds = obj_embeds.masked_fill(obj_sem_masks.unsqueeze(2).unsqueeze(3).logical_not(), 0.0)
    #     obj_sem_mask_embeds = self.sem_mask_embeddings(
    #         torch.zeros((batch_size, num_objs)).long().cuda()
    #     ).unsqueeze(2) * obj_sem_masks.logical_not().unsqueeze(2).unsqueeze(3)
    #     obj_embeds = obj_embeds + obj_sem_mask_embeds

    #     obj_embeds_pre = obj_embeds

    #     # 记录预嵌入
    #     # 注意：我们有三类嵌入，分别是从 PointNet 提取的原始嵌入、经过 token 化的预嵌入，以及 transformer 处理后的最终嵌入

    #     for i, pc_layer in enumerate(self.spatial_encoder):
    #         query_pos = self.loc_layers[0](obj_locs)  # 不取平均，保持视角维度
    #         if is_train:
    #             obj_embeds = obj_embeds + query_pos * query_pos_mask
    #         else:
    #             obj_embeds = obj_embeds + query_pos
        
    #         obj_embeds, self_attn_matrices = pc_layer(
    #             obj_embeds.permute(0, 2, 1, 3).reshape(batch_size * view_num, num_objs, -1),  # 展平后输入 transformer
    #             pairwise_locs.reshape(batch_size * view_num, num_objs,num_objs, -1), 
    #             tgt_key_padding_mask=obj_masks.unsqueeze(1).expand(-1, view_num, -1).logical_not().reshape(batch_size * view_num, -1)
    #         )
    #         # 还原嵌入维度
    #         obj_embeds = obj_embeds.reshape(batch_size, view_num, num_objs, -1).permute(0, 2, 1, 3)

    #     return obj_embeds, obj_embeds_pre, obj_sem_cls


    def forward(self, obj_pcds, obj_locs, obj_masks, obj_sem_masks, obj_labels=None, cur_step=None, max_steps=None, is_train=False):
        if self.freeze_feature:
            self.freeze_bn(self.point_feature_extractor)
        # get obj_embdes
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.point_feature_extractor(einops.rearrange(obj_pcds, 'b o p d -> (b o) p d') )
        obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        obj_embeds = self.dropout(obj_embeds)
        if self.freeze_feature:
            obj_embeds = obj_embeds.detach()
        
        # get semantic cls embeds
        obj_sem_cls = self.point_cls_head(obj_embeds) # B, O, 607
        if self.freeze_feature:
            obj_sem_cls = obj_sem_cls.detach()
        if self.mixup_strategy != None:
            obj_sem_cls_mix = self.mixup_function(obj_sem_cls, obj_labels, cur_step, max_steps)
        else:
            obj_sem_cls_mix = obj_sem_cls.clone()
        obj_sem_cls_mix = torch.argmax(obj_sem_cls_mix, dim=2)
        obj_sem_cls_embeds = torch.Tensor([self.cat2vec[self.int2cat[int(i)]] for i in obj_sem_cls_mix.view(batch_size * num_objs)])
        obj_sem_cls_embeds = obj_sem_cls_embeds.view(batch_size, num_objs, 300).cuda()
        obj_sem_cls_embeds = self.sem_cls_embed_layer(obj_sem_cls_embeds)
        if is_train:
            # print(obj_masks.shape)
            # obj_embeds_mask = generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10)
            # obj_sem_cls_embeds_mask = generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10)
            # query_pos_mask = generate_spatial_feature_mask(target_id, obj_masks, pairwise_locs, mask_prob=0.15, max_dist=10)
            # print(obj_embeds_mask)
            # obj_embeds = obj_embeds * obj_embeds_mask + obj_sem_cls_embeds * obj_sem_cls_embeds_mask

            # 随机mask
            obj_embeds_mask = generate_feature_mask(obj_masks, obj_embeds.shape[-1], mask_prob=0.05)
            obj_sem_cls_embeds_mask = generate_feature_mask(obj_masks, obj_embeds.shape[-1], mask_prob=0.1)
            query_pos_mask = generate_feature_mask(obj_masks, obj_embeds.shape[-1], mask_prob=0.1)
            obj_embeds = obj_embeds * obj_embeds_mask + obj_sem_cls_embeds * obj_sem_cls_embeds_mask

            # obj_embeds_no_mask = obj_embeds + obj_sem_cls_embeds
        else:
            # obj_embeds_no_mask = obj_embeds + obj_sem_cls_embeds
            obj_embeds = obj_embeds + obj_sem_cls_embeds

        # obj_embeds_no_mask = obj_embeds + obj_sem_cls_embeds
        # obj_embeds = obj_embeds + obj_sem_cls_embeds
        

        
        # get semantic mask embeds
        obj_embeds = obj_embeds.masked_fill(obj_sem_masks.unsqueeze(2).logical_not(), 0.0)
        obj_sem_mask_embeds = self.sem_mask_embeddings(torch.zeros((batch_size, num_objs)).long().cuda()) * obj_sem_masks.logical_not().unsqueeze(2)
        obj_embeds = obj_embeds + obj_sem_mask_embeds

        # obj_embeds_no_mask = obj_embeds_no_mask.masked_fill(obj_sem_masks.unsqueeze(2).logical_not(), 0.0)
        #obj_embeds_no_mask = obj_embeds_no_mask + obj_sem_mask_embeds
        
        # record pre embedes
        # note: in our implementation, there are three types of embds, raw embeds from PointNet, pre embeds after tokenization, post embeds after transformers
        obj_embeds_pre = obj_embeds
        
        # spatial reasoning
        pairwise_locs = calc_pairwise_locs(obj_locs[:, :, :3], obj_locs[:, :, 3:], pairwise_rel_type=self.pairwise_rel_type, spatial_dist_norm=True, spatial_dim=self.spatial_dim)

        for i , pc_layer in enumerate(self.spatial_encoder):
            query_pos = self.loc_layers[0](obj_locs)
            if is_train:
                obj_embeds = obj_embeds + query_pos * query_pos_mask
                # obj_embeds = obj_embeds + query_pos
            else:
                # spacial_obj_embeds = obj_embeds + query_pos
                obj_embeds = obj_embeds + query_pos

            # spacial_obj_embeds = obj_embeds + query_pos
            # obj_embeds = obj_embeds + query_pos

            obj_embeds, self_attn_matrices = pc_layer(obj_embeds, pairwise_locs, tgt_key_padding_mask=obj_masks.logical_not())
            # spacial_obj_embeds, self_attn_matrices = pc_layer(spacial_obj_embeds, pairwise_locs, tgt_key_padding_mask=obj_masks.logical_not())

        
        return obj_embeds, obj_embeds_pre, obj_sem_cls




if __name__ == '__main__':
    x = PointTokenizeEncoder(backbone='pointnet++', hidden_size=768, path="project/pretrain_weights/pointnet_tokenizer.pth", freeze_feature=True).cuda()
    obj_pcds = torch.ones((10, 10, 1024, 6)).float().cuda()
    obj_locs = torch.ones((10, 10, 6)).cuda()
    obj_masks = torch.ones((10, 10)).cuda()
    obj_sem_masks = torch.ones((10, 10)).cuda()
    x(obj_pcds, obj_locs, obj_masks, obj_sem_masks)
    
    
    
    