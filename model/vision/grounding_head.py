import torch
import torch.nn as nn

import model.vision.pointnet2.pointnet2_utils as pointnet2_utils
from model.vision.basic_modules import get_mlp_head
from pipeline.registry import registry

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

@registry.register_other_model("ground_head_v1")
class GroundHeadV1(nn.Module):
    def __init__(self, input_size=768, hidden_size=768, sem_cls_size=607, dropout=0.3):
        super().__init__()
        self.og3d_head = get_mlp_head(
            input_size, hidden_size, 
            1, dropout=dropout
        )
        self.txt_clf_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_head = get_mlp_head(
            input_size, hidden_size, 
            sem_cls_size, dropout=dropout
        )
        self.obj3d_clf_pre_head = get_mlp_head(
            input_size, hidden_size,
            sem_cls_size, dropout=dropout
        )
        
    def forward(self, txt_embeds_1, obj_embeds_1, obj_pre_embeds, obj_masks):
        # print(txt_embeds.shape, obj_embeds.shape, obj_pre_embeds.shape, obj_masks.shape)
        # torch.Size([47, 100, 768]) torch.Size([47, 80, 768]) torch.Size([47, 80, 768]) torch.Size([47, 80])
        og3d_logits_1 = self.og3d_head(obj_embeds_1).squeeze(2)
        # og3d_logits_2 = self.og3d_head(obj_embeds_2).squeeze(2)
        og3d_logits = og3d_logits_1
        # aggr_text_feat: [47, 768] -> 添加一个维度变为 [47, 1, 768]
        # aggr_text_feat_expanded = aggr_text_feat.unsqueeze(1).expand(-1, 80, -1)  # [47, 80, 768]  # [47, 80, 768]
        # feat_concat = torch.cat((aggr_text_feat_expanded, obj_embeds), dim=-1)  # [47, 80, 768]
        # og3d_logits = self.og3d_head(feat_concat).squeeze(2)
        og3d_logits = og3d_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        txt_embeds_1 = txt_embeds_1.detach()
        #txt_embeds_2 = txt_embeds_2.detach()

        obj_embeds_1 = obj_embeds_1.detach()
        #obj_embeds_2 = obj_embeds_2.detach()

        obj_pre_embeds = obj_pre_embeds.detach()
        
        txt_cls_logits = self.txt_clf_head(txt_embeds_1[:, 0])
        obj_cls_logits = self.obj3d_clf_head(obj_embeds_1) 
        obj_cls_pre_logits = self.obj3d_clf_pre_head(obj_pre_embeds)
        return txt_cls_logits, obj_cls_logits, obj_cls_pre_logits, og3d_logits
    
if __name__ == '__main__':
    GroundHeadV1()