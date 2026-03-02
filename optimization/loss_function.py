import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pipeline.registry import registry
from torch.autograd import Variable


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities

class CCL(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, tau=0.05, method='gce', q=0.25, ratio=0):
        super(CCL, self).__init__()
        self.tau = tau
        self.method = method
        self.q = q  # Default value of q
        self.ratio = ratio

    def forward(self, scores):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        randn, eye = torch.rand_like(scores), torch.eye(scores.shape[0]).cuda()
        randn[eye > 0] = randn.min(dim=1)[0] - 1
        n = scores.shape[0]
        num = n - 1 if self.ratio <= 0 or self.ratio >= 1 else int(self.ratio * n)
        V, K = randn.topk(num, dim=1)
        mask = torch.zeros_like(scores)
        mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

        # Dynamic calculation of q based on scores
        # For each i,j pair, use scores[i, j] to compute q
        dynamic_q = scores  # Here q becomes a tensor the same size as scores, each element will be used as q for the corresponding loss calculation
        
        # Define the criterion function with q
        criterion = lambda x, q: ((1. - (1. - x + eps) ** q) / q * mask).sum(1).mean()

        loss_i2t = criterion(i2t, i2t)
        loss_t2i = criterion(t2i, t2i)

        return loss_i2t + loss_t2i

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

@registry.register_optimizer("refer_loss_v1")
def get_refer_loss_v1(txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, og3d_logits, tgt_object_label, tgt_object_id, obj_labels, obj_masks, point_feat_1, text_feat_1, fusion_point_feat_1, fusion_text_feat_1):
    og3d_loss = F.cross_entropy(og3d_logits, tgt_object_id.squeeze(1))
    txt_cls_loss = F.cross_entropy(txt_cls_logits, tgt_object_label.squeeze(1))
    
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()

    ccl_criterion = CCL()

    point_feat_1 = l2norm(point_feat_1, dim=-1)
    text_feat_1 = l2norm(text_feat_1, dim=-1)

    fusion_point_feat_1 = l2norm(fusion_point_feat_1, dim=-1)

    fusion_text_feat_1 = l2norm(fusion_text_feat_1, dim=-1)

    score_1 = torch.matmul(point_feat_1, text_feat_1.T) 
    
    matching_loss_1 = ccl_criterion(score_1)
    matching_loss = matching_loss_1 

    total_loss = (obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss) + 1 *  matching_loss + 1 * og3d_loss 
    return total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, matching_loss, matching_loss


@registry.register_optimizer("qa_loss_v1")
def get_qa_loss_v1(txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, og3d_logits, answer_scores, tgt_object_label, tgt_object_id, obj_labels, obj_masks, answer_label):
    og3d_logits = og3d_logits.masked_fill_(og3d_logits == -float('inf'), 0)
    og3d_loss = F.binary_cross_entropy_with_logits(og3d_logits, tgt_object_id.float(), reduction='sum', weight=obj_masks) / float(tgt_object_id.shape[0])
    txt_cls_loss = F.binary_cross_entropy_with_logits(txt_cls_logits, tgt_object_label.float(), reduction='sum') / float(tgt_object_label.shape[0])
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    answer_loss = F.binary_cross_entropy_with_logits(answer_scores, answer_label.float(), reduction='sum') / answer_scores.shape[0]
    total_loss = og3d_loss + txt_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss + answer_loss
    return total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss

@registry.register_optimizer("pretrain_loss_v1")
def get_pretrain_loss_v1(txt_lm_cls_logits, masked_lm_labels, scene_txt_match_logits, replace, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, obj_labels, obj_sem_masks, obj_masks):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    masked_lm_labels.masked_fill_(replace.unsqueeze(1), -1)
    lm_cls_loss = loss_fct(txt_lm_cls_logits.permute(0, 2, 1), masked_lm_labels)
    match_loss = loss_fct(scene_txt_match_logits, replace.long())
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss_mask = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks.logical_not()).sum() / (obj_masks * obj_sem_masks.logical_not()).sum()
    obj_cls_pre_loss_unmask = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks).sum() / (obj_masks * obj_sem_masks).sum()
    obj_cls_post_loss_mask = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks.logical_not()).sum() / (obj_masks * obj_sem_masks.logical_not()).sum()
    obj_cls_post_loss_unmask = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks * obj_sem_masks).sum() / (obj_masks * obj_sem_masks).sum()
    total_loss = lm_cls_loss + match_loss + obj_cls_raw_loss + obj_cls_pre_loss_unmask + obj_cls_post_loss
    return total_loss, lm_cls_loss, match_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, obj_cls_pre_loss_mask, obj_cls_pre_loss_unmask, obj_cls_post_loss_mask, obj_cls_post_loss_unmask

@registry.register_optimizer("sqa_loss_v1")
def get_qa_loss_v1(obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, answer_scores, obj_labels, obj_masks, answer_label):
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    answer_loss = F.binary_cross_entropy_with_logits(answer_scores, answer_label.float(), reduction='sum') / answer_scores.shape[0]
    total_loss = obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss + answer_loss
    return total_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss


@registry.register_optimizer("caption_loss_v1")
def get_caption_loss_v1(txt_lm_cls_logits, masked_lm_labels, obj_cls_post_logits, obj_cls_pre_logits, obj_cls_raw_logits, obj_labels, obj_masks):
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    lm_cls_loss = loss_fct(txt_lm_cls_logits.permute(0, 2, 1), masked_lm_labels)
    obj_cls_raw_loss = (F.cross_entropy(obj_cls_raw_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_pre_loss = (F.cross_entropy(obj_cls_pre_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    obj_cls_post_loss = (F.cross_entropy(obj_cls_post_logits.permute(0, 2, 1), obj_labels, reduction='none') * obj_masks).sum() / obj_masks.sum()
    total_loss = lm_cls_loss + obj_cls_raw_loss + obj_cls_pre_loss + obj_cls_post_loss
    return total_loss, lm_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss