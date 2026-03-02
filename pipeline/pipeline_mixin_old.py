import torch
import math
from tqdm import tqdm
import json
import time
import numpy as np
from utils.cider.cider import Cider
from utils.bleu.bleu import Bleu
from utils.meteor.meteor import Meteor
from utils.rouge.rouge import Rouge
from utils.caption_search import GreedySearch
from utils.box_util import get_3d_box

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class NormalDataloaderMixin:
    def __init__(self) -> None:
        pass
    
    def build_dataloader(self, dataset):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size, 
            num_workers=8,
            pin_memory=True,
            shuffle=True, 
            drop_last = True)
        return data_loader

    def build_train_test_dataloader(self):
        if hasattr(self, 'train_dataset'):
            self.train_data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=8,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )
        if hasattr(self, 'test_dataset'):
            self.test_data_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=4,
                num_workers=8,
                pin_memory=True,
                shuffle=False
            )
     
    def prepare_data(self, data_dict):    
        for key in data_dict:
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].cuda()
                
class ModelOptimizationMixin(object):
    def __init__(self):
        pass
    
    @staticmethod
    def warmup_cosine(step, warmup_step, tot_step):
        if step <= warmup_step:
            return step / warmup_step
        return max(0.5 * (1 + math.cos((step - warmup_step) / (tot_step - warmup_step) * math.pi)), 1e-5)
    
    def no_decay_param_group(self, parameters, lr):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        decay_params = []
        no_decay_params = []
        for n, p in parameters:
            if p.requires_grad == False:
                continue
            if not any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optimizer_grouped_parameters = [
            {'params': decay_params,
            'weight_decay': 0.01, 'lr': lr},
            {'params': no_decay_params,
            'weight_decay': 0.0, 'lr': lr}
        ]
        return optimizer_grouped_parameters

def l2norm(X, dim, eps=1e-8):
    """
    L2-normalize along the specified axis of X
    """
    norm = np.sqrt(np.sum(np.power(X, 2), axis=dim, keepdims=True)) + eps
    X = X / norm
    return X

def calculate_recall_at_k(similarity_matrix, text_scene, point_scene, k_values):
    """
    计算文本检索点云的 Recall@K，同时返回每个文本的 top-1、top-5 和 top-10 结果
    """
    recalls = {f'Recall@{k}': [] for k in k_values}
    n_texts = similarity_matrix.shape[0]  # 文本数量
    top_1_results = []  # 保存每个文本的 top-1 匹配场景
    top_5_results = []  # 保存每个文本的 top-5 匹配场景
    top_10_results = []  # 保存每个文本的 top-10 匹配场景

    for i in tqdm(range(n_texts)):
        # 1. 获取第 i 个文本样本的相似度排序
        top_indices = np.argsort(-similarity_matrix[i])  # 按相似度降序排序
        top_scene_names = [point_scene[idx] for idx in top_indices]  # 排序后的场景标签

        # 保存 top-1、top-5 和 top-10 场景
        top_1_results.append(top_scene_names[0])
        top_5_results.append(top_scene_names[:5])
        top_10_results.append(top_scene_names[:10])

        # 2. 针对每个 k 计算是否命中
        for k in k_values:
            if text_scene[i] in top_scene_names[:k]:
                recalls[f'Recall@{k}'].append(1)  # 命中
            else:
                recalls[f'Recall@{k}'].append(0)  # 未命中

    # 3. 计算平均 Recall@k
    recalls = {key: np.mean(values) for key, values in recalls.items()}
    return recalls, top_1_results, top_5_results, top_10_results

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def calculate_recall_at_k_v2(similarity_matrix, text_scene, point_scene, k_values, challenge, prompt): 
    """
    Calculate the Recall@K of the text retrieval point cloud, and return the top-1, top-5, and top-10 results for each text, categorizing and statistically analyzing them by challenge and prompt.
    """
    recalls = {f'Recall@{k}': [] for k in k_values}
    n_texts = similarity_matrix.shape[0]  
    top_1_results = []
    top_5_results = []
    top_10_results = []

    challenge_recalls = {ch: {f'Recall@{k}': [] for k in k_values} for ch in set(challenge)}
    prompt_recalls = {pr: {f'Recall@{k}': [] for k in k_values} for pr in set(prompt)}

    for i in tqdm(range(n_texts)):
        top_indices = np.argsort(-similarity_matrix[i]) 
        top_scene_names = [point_scene[idx] for idx in top_indices]

        top_1_results.append(top_scene_names[0])
        top_5_results.append(top_scene_names[:5])
        top_10_results.append(top_scene_names[:10])

        ch = challenge[i]
        pr = prompt[i]

        for k in k_values:
            hit = 1 if text_scene[i] in top_scene_names[:k] else 0
            recalls[f'Recall@{k}'].append(hit) 
            challenge_recalls[ch][f'Recall@{k}'].append(hit)
            prompt_recalls[pr][f'Recall@{k}'].append(hit)

    recalls = {key: np.mean(values) for key, values in recalls.items()}

    for ch, recs in challenge_recalls.items():
        challenge_recalls[ch] = {key: np.mean(values) for key, values in recs.items()}
        print(f"Challenge={ch} Recall: {challenge_recalls[ch]}")

    for pr, recs in prompt_recalls.items():
        prompt_recalls[pr] = {key: np.mean(values) for key, values in recs.items()}
        print(f"Prompt={pr} Recall: {prompt_recalls[pr]}")

    return recalls, top_1_results, top_5_results, top_10_results

class ModelEvaluationMixin(object):
    ''' scanrefer '''
    # def eval_matching_grounding(self, epoch):
    #     print("Eval matching and grounding")
    #     text_feat_1 = []
    #     text_feat_2 = []
    #     is_multi = []
    #     text_scene = []
    #     text_mask = []
    #     text_target_id_25 = []
    #     text_target_id_50 = []

    #     no_aggr_point_feat = []
    #     no_aggr_text_feat = []
    #     point_feat_1 = []
    #     point_feat_2 = []
    #     point_scene = []

    #     obj_locs = []
    #     obj_masks = []

    #     added_scenes = set() 

    #     ''' Encode data '''
    #     start_time = time.time()
    #     for i, data_dict in enumerate(tqdm(self.test_data_loader)):
    
    #         self.prepare_data(data_dict)
    #         if 'cur_step' not in data_dict.keys():
    #             data_dict['cur_step'] = 1
    #             data_dict['total_steps'] = 1


    #         lang_basic_features = self.lang_encoder(data_dict['txt_ids'], data_dict['txt_masks']).last_hidden_state
    #         # lang_matching_features = self.text_mapping(lang_basic_features)

    #         # lang_basic_features = self.text_mapping(lang_basic_features)
    #         # lang_basic_features_2 = self.text_mapping_2(lang_basic_features)

    #         indices = torch.arange(data_dict['txt_masks'].size(1), device=data_dict['txt_masks'].device).expand_as(data_dict['txt_masks'])
    #         masked_indices = torch.where(data_dict['txt_masks'] == 1, indices, torch.tensor(-1, device=data_dict['txt_masks'].device))
    #         last_indices = masked_indices.max(dim=1).values + 1
 
    #         aggr_text_feat_1 = self.text_fusion_head(lang_basic_features, last_indices)
    #         # aggr_text_feat_2 = self.text_fusion_head(lang_matching_features_2, data_dict['txt_masks'])

    #         for idx in range(aggr_text_feat_1.shape[0]):

    #             text_feat_1.append(aggr_text_feat_1[idx].cpu().detach().numpy())
    #             # text_feat_2.append(aggr_text_feat_2[idx].cpu().detach().numpy())
    #             no_aggr_text_feat.append(lang_basic_features[idx].cpu().detach().numpy())
    #             text_scene.append(data_dict["scan_id"][idx])
    #             is_multi.append(data_dict["is_multiple"][idx])
    #             # text_target_id.append(data_dict["tgt_object_id"][idx].squeeze(-1).item())
    #             # text_target_id.append(data_dict["tgt_object_id"][idx].cpu().detach().numpy())
    #             result = np.where(data_dict["tgt_object_id_iou25"][idx].cpu().detach().numpy() == 1)[0]
    #             text_target_id_25.append(result)
    #             result_ = np.where(data_dict["tgt_object_id_iou50"][idx].cpu().detach().numpy() == 1)[0]
    #             text_target_id_50.append(result_)
    #             text_mask.append(data_dict['txt_masks'][idx].cpu().detach().numpy())

    #         ori_obj_fts = data_dict['obj_fts']
    #         ori_obj_locs = data_dict['obj_locs']
    #         ori_obj_masks = data_dict['obj_masks']
    #         ori_obj_sem_masks = data_dict['obj_sem_masks']
    #         ori_obj_labels = data_dict['obj_labels']

    #         exp_obj_locs = data_dict['obj_locs']
    #         exp_obj_masks = data_dict['obj_masks']

    #         point_basic_features_1, _, _ = self.point_encoder(
    #             ori_obj_fts, ori_obj_locs, ori_obj_masks, ori_obj_sem_masks,
    #             ori_obj_labels, data_dict['cur_step'], data_dict['total_steps'], False
    #         )
    #         # point_basic_features_1 = self.point_mapping(point_basic_features_1)
    #         # point_basic_features_2 = self.point_mapping_2(point_basic_features_2)

    #         indices = torch.arange(data_dict['obj_masks'].size(1), device=data_dict['obj_masks'].device).expand_as(data_dict['obj_masks'])
    #         masked_indices = torch.where(data_dict['obj_masks'] == 1, indices, torch.tensor(-1, device=data_dict['obj_masks'].device))
    #         last_indices = masked_indices.max(dim=1).values + 1
            
    #         aggr_point_feat_1 = self.point_fusion_head(point_basic_features_1, last_indices)
    #         # aggr_point_feat_2 = self.point_fusion_head(point_basic_features_1, ori_obj_masks)
            
            
    #         for idx, scan_id in enumerate(data_dict["scan_id"]):
    #             if scan_id not in added_scenes:
    #                 added_scenes.add(scan_id)
    #                 point_scene.append(scan_id)
    #                 point_feat_1.append(aggr_point_feat_1[idx].cpu().detach().numpy())
    #                 # point_feat_2.append(aggr_point_feat_2[idx].cpu().detach().numpy())
    #                 no_aggr_point_feat.append(point_basic_features_1[idx].cpu().detach().numpy())
    #                 obj_locs.append(exp_obj_locs[idx].cpu().detach().numpy())
    #                 obj_masks.append(exp_obj_masks[idx].cpu().detach().numpy())


    #     text_feat_1 = l2norm(np.array(text_feat_1), dim=-1)
    #     # text_feat_2 = l2norm(np.array(text_feat_2), dim=-1)
    #     point_feat_1 = l2norm(np.array(point_feat_1), dim=-1)
    #     # point_feat_2 = l2norm(np.array(point_feat_2), dim=-1)


    #     similarity_matrix_1 = np.matmul(text_feat_1, point_feat_1.T) # 8000 x 141
    #     # similarity_matrix_2 = np.matmul(text_feat_2, point_feat_2.T)
    #     similarity_matrix = similarity_matrix_1 # + similarity_matrix_2    # similarity_matrix_2 比较好

    #     # Matching
    #     k_values = [1, 5, 10]
    #     recalls, top_1_results, top_5_results, top_10_results = calculate_recall_at_k(similarity_matrix, text_scene, point_scene, k_values)
    #     print("Recall Results:")
    #     for k, value in recalls.items():
    #         print(f"{k}: {value:.4f}")

    #     # Grounding 
    #     scene_to_point_idx = {scene: idx for idx, scene in enumerate(point_scene)}

    #     correct_count_top1_25 = 0
    #     correct_count_top1_50 = 0

    #     correct_count_top1_50_multi = 0
    #     correct_count_top1_50_uni = 0
    #     correct_count_top1_25_multi = 0
    #     correct_count_top1_25_uni = 0

    #     total_count_top1 = 0
    #     total_count_top1_multi = 0
    #     total_count_top1_uni = 0

    #     batch_size = 8
    #     num_batches = len(text_scene) // batch_size

    #     for i in tqdm(range(num_batches)):
    #         start = i * batch_size
    #         end = start + batch_size

    #         top1_text_feat = []
    #         top1_text_mask = []
    #         top1_target_id_25 = []
    #         top1_target_id_50 = []
    #         top1_point_feat = []
    #         top1_obj_locs = []
    #         top1_obj_masks = []
    #         top1_scene_ids = []
    #         top1_text_scene_ids = []
    #         top1_is_multi = []


    #         for idx in range(start, end):
    #             top_1_scene = top_1_results[idx]
    #             if top_1_scene in scene_to_point_idx:
    #                 point_idx = scene_to_point_idx[top_1_scene]
    #                 top1_text_feat.append(no_aggr_text_feat[idx])
    #                 top1_text_mask.append(text_mask[idx])
    #                 top1_target_id_25.append(text_target_id_25[idx])
    #                 top1_target_id_50.append(text_target_id_50[idx])
    #                 top1_point_feat.append(no_aggr_point_feat[point_idx])
    #                 top1_obj_locs.append(obj_locs[point_idx])
    #                 top1_obj_masks.append(obj_masks[point_idx])
    #                 top1_scene_ids.append(top_1_scene)
    #                 top1_text_scene_ids.append(text_scene[idx])
    #                 top1_is_multi.append(is_multi[idx])

    #         def to_tensor(data):
    #             return torch.tensor(np.array(data)).to('cuda')

    #         top1_text_feat = to_tensor(top1_text_feat)
    #         top1_text_mask = to_tensor(top1_text_mask)
    #         top1_point_feat = to_tensor(top1_point_feat)
    #         top1_obj_locs = to_tensor(top1_obj_locs)
    #         top1_obj_masks = to_tensor(top1_obj_masks)

    #         # Top 1 Grounding
    #         if len(top1_text_feat) > 0:
    #             _, point_fuse_feature = self.unified_encoder(
    #                 top1_text_feat, top1_text_mask, top1_point_feat, top1_obj_locs, top1_obj_masks
    #             )
    #             pro = self.ground_head.og3d_head(point_fuse_feature)
    #             max_pro_idx = pro.argmax(dim=1)

    #             for idx in range(len(max_pro_idx)):
    #                 if max_pro_idx[idx].item() in top1_target_id_25[idx]:
    #                     if top1_scene_ids[idx] == top1_text_scene_ids[idx]:
    #                         correct_count_top1_25 += 1
    #                         if top1_is_multi[idx] == True:
    #                             correct_count_top1_25_multi += 1
    #                         else:
    #                             correct_count_top1_25_uni += 1
    #                 if max_pro_idx[idx].item() in top1_target_id_50[idx]:
    #                     if top1_scene_ids[idx] == top1_text_scene_ids[idx]:
    #                         correct_count_top1_50 += 1
    #                         if top1_is_multi[idx] == True:
    #                             correct_count_top1_50_multi += 1
    #                         else:
    #                             correct_count_top1_50_uni += 1
    #                 total_count_top1 += 1
    #                 if top1_is_multi[idx] == True:
    #                     total_count_top1_multi += 1
    #                 else:
    #                     total_count_top1_uni += 1
    #     end_time = time.time()
    #     elapsed_time = end_time - start_time
    #     print(f"Time: {elapsed_time} 秒")
    #     accuracy_top1 = correct_count_top1_25 / total_count_top1
    #     accuracy_top1_ = correct_count_top1_50 / total_count_top1

    #     print(f"Top 1 - @25 Correct: {correct_count_top1_25}/{total_count_top1}, Accuracy: {accuracy_top1 * 100:.2f}%")
    #     print(f"Top 1 - @50 Correct: {correct_count_top1_50}/{total_count_top1}, Accuracy: {accuracy_top1_ * 100:.2f}%")

        # print(f"Top 1 - @25 Multi:  {correct_count_top1_25_multi}/{total_count_top1_multi}, Accuracy: {(correct_count_top1_25_multi/total_count_top1_multi) * 100:.2f}%")
        # print(f"Top 1 - @25 Uni:  {correct_count_top1_25_uni}/{total_count_top1_uni}, Accuracy: {(correct_count_top1_25_uni/total_count_top1_uni) * 100:.2f}%")

        # print(f"Top 1 - @50 Multi:  {correct_count_top1_50_multi}/{total_count_top1_multi}, Accuracy: {(correct_count_top1_50_multi/total_count_top1_multi) * 100:.2f}%")
        # print(f"Top 1 - @50 Uni:  {correct_count_top1_50_uni}/{total_count_top1_uni}, Accuracy: {(correct_count_top1_50_uni/total_count_top1_uni) * 100:.2f}%")
    
    # ''' cross_scene '''
    def eval_matching_grounding(self, epoch):
        print("Eval matching and grounding")
        text_feat_1 = []
        text_feat_2 = []
        text_scene = []
        text_mask = []
        text_target_id_25 = []
        text_target_id_50 = []

        no_aggr_point_feat = []
        no_aggr_text_feat = []
        point_feat_1 = []
        point_feat_2 = []
        point_scene = []
        challenge = []
        prompt = []

        obj_locs = []
        obj_masks = []

        added_scenes = set()

        ''' Encode data '''
        for i, data_dict in enumerate(tqdm(self.test_data_loader)):
            self.prepare_data(data_dict)
            if 'cur_step' not in data_dict.keys():
                data_dict['cur_step'] = 1
                data_dict['total_steps'] = 1

            lang_basic_features = self.lang_encoder(data_dict['txt_ids'], data_dict['txt_masks']).last_hidden_state
 
            indices = torch.arange(data_dict['txt_masks'].size(1), device=data_dict['txt_masks'].device).expand_as(data_dict['txt_masks'])
            masked_indices = torch.where(data_dict['txt_masks'] == 1, indices, torch.tensor(-1, device=data_dict['txt_masks'].device))
            last_indices = masked_indices.max(dim=1).values + 1
 
            aggr_text_feat_1 = self.text_fusion_head(lang_basic_features, last_indices)

            for idx in range(aggr_text_feat_1.shape[0]):

                text_feat_1.append(aggr_text_feat_1[idx].cpu().detach().numpy())
                no_aggr_text_feat.append(lang_basic_features[idx].cpu().detach().numpy())
                text_scene.append(data_dict["scan_id"][idx])
                challenge.append(data_dict["challenge"][idx])
                prompt.append(data_dict["prompt"][idx])
                result = np.where(data_dict["tgt_object_id_iou25"][idx].cpu().detach().numpy() == 1)[0]
                text_target_id_25.append(result)
                result_ = np.where(data_dict["tgt_object_id_iou50"][idx].cpu().detach().numpy() == 1)[0]
                text_target_id_50.append(result_)
                text_mask.append(data_dict['txt_masks'][idx].cpu().detach().numpy())

            ori_obj_fts = data_dict['obj_fts']
            ori_obj_locs = data_dict['obj_locs']
            ori_obj_masks = data_dict['obj_masks']
            ori_obj_sem_masks = data_dict['obj_sem_masks']
            ori_obj_labels = data_dict['obj_labels']

            exp_obj_locs = data_dict['obj_locs']
            exp_obj_masks = data_dict['obj_masks']

            point_basic_features_1, _, _ = self.point_encoder(
                ori_obj_fts, ori_obj_locs, ori_obj_masks, ori_obj_sem_masks,
                ori_obj_labels, data_dict['cur_step'], data_dict['total_steps'], False
            )

            indices = torch.arange(data_dict['obj_masks'].size(1), device=data_dict['obj_masks'].device).expand_as(data_dict['obj_masks'])
            masked_indices = torch.where(data_dict['obj_masks'] == 1, indices, torch.tensor(-1, device=data_dict['obj_masks'].device))
            last_indices = masked_indices.max(dim=1).values + 1
            
            aggr_point_feat_1 = self.point_fusion_head(point_basic_features_1, last_indices)

            for idx, scan_id in enumerate(data_dict["scan_id"]):
                if scan_id not in added_scenes:
                    added_scenes.add(scan_id)
                    point_scene.append(scan_id)
                    point_feat_1.append(aggr_point_feat_1[idx].cpu().detach().numpy())
                    no_aggr_point_feat.append(point_basic_features_1[idx].cpu().detach().numpy())
                    obj_locs.append(exp_obj_locs[idx].cpu().detach().numpy())
                    obj_masks.append(exp_obj_masks[idx].cpu().detach().numpy())


        text_feat_1 = l2norm(np.array(text_feat_1), dim=-1)
        point_feat_1 = l2norm(np.array(point_feat_1), dim=-1)

        similarity_matrix_1 = np.matmul(text_feat_1, point_feat_1.T) # 8000 x 141
        similarity_matrix = similarity_matrix_1 

        # Matching
        k_values = [1, 5, 10]
        # recalls, top_1_results, top_5_results, top_10_results = calculate_recall_at_k(similarity_matrix, text_scene, point_scene, k_values, challenge, prompt)
        recalls, top_1_results, top_5_results, top_10_results = calculate_recall_at_k_v2(similarity_matrix, text_scene, point_scene, k_values, challenge, prompt)
        print("Recall Results:")
        co= 0
        for k, value in recalls.items():
            print(f"{k}: {value:.4f}")

        # Grounding 
        scene_to_point_idx = {scene: idx for idx, scene in enumerate(point_scene)}

        correct_count_top1_25 = 0
        correct_count_top1_50 = 0
        total_count_top1 = 0

        correct_count_top1_conspicuous_25 = 0
        correct_count_top1_regular_25 = 0
        correct_count_top1_confusing_25 = 0

        correct_count_top1_conspicuous_50 = 0
        correct_count_top1_regular_50 = 0
        correct_count_top1_confusing_50 = 0

        total_count_top1_conspicuous = 0
        total_count_top1_regular = 0 
        total_count_top1_confusing = 0

        correct_count_top1_discriminative_25 = 0
        correct_count_top1_spacial_25 = 0
        correct_count_top1_comprehensive_25 = 0
        correct_count_top1_fuzzy_25 = 0

        correct_count_top1_discriminative_50 = 0
        correct_count_top1_spacial_50 = 0
        correct_count_top1_comprehensive_50 = 0
        correct_count_top1_fuzzy_50 = 0

        total_count_top1_discriminative = 0
        total_count_top1_spacial = 0
        total_count_top1_comprehensive = 0
        total_count_top1_fuzzy = 0

        batch_size = 3
        num_batches = len(text_scene) // batch_size

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size

            # Top 1 
            top1_text_feat = []
            top1_text_mask = []
            top1_target_id_25 = []
            top1_target_id_50 = []
            top1_point_feat = []
            top1_obj_locs = []
            top1_obj_masks = []
            top1_scene_ids = []
            top1_text_scene_ids = []

            top1_prompt_ids = []
            top1_challenge_ids = []


            for idx in range(start, end):
                top_1_scene = top_1_results[idx]
                if top_1_scene in scene_to_point_idx:
                    point_idx = scene_to_point_idx[top_1_scene]
                    top1_text_feat.append(no_aggr_text_feat[idx])
                    top1_text_mask.append(text_mask[idx])
                    top1_target_id_25.append(text_target_id_25[idx])
                    top1_target_id_50.append(text_target_id_50[idx])
                    top1_point_feat.append(no_aggr_point_feat[point_idx])
                    top1_obj_locs.append(obj_locs[point_idx])
                    top1_obj_masks.append(obj_masks[point_idx])
                    top1_scene_ids.append(top_1_scene)
                    top1_text_scene_ids.append(text_scene[idx])
                    top1_prompt_ids.append(prompt[idx])
                    top1_challenge_ids.append(challenge[idx])
            def to_tensor(data):
                return torch.tensor(np.array(data)).to('cuda')

            top1_text_feat = to_tensor(top1_text_feat)
            top1_text_mask = to_tensor(top1_text_mask)
            top1_point_feat = to_tensor(top1_point_feat)
            top1_obj_locs = to_tensor(top1_obj_locs)
            top1_obj_masks = to_tensor(top1_obj_masks)

            # Top 1 Grounding
            if len(top1_text_feat) > 0:
                _, point_fuse_feature = self.unified_encoder(
                    top1_text_feat, top1_text_mask, top1_point_feat, top1_obj_locs, top1_obj_masks
                )
                pro = self.ground_head.og3d_head(point_fuse_feature)
                max_pro_idx = pro.argmax(dim=1)

                for idx in range(len(max_pro_idx)):
                    if max_pro_idx[idx].item() in top1_target_id_25[idx]:
                        if top1_scene_ids[idx] == top1_text_scene_ids[idx]:
                            if top1_challenge_ids[idx] == 'conspicuous':
                                correct_count_top1_conspicuous_25 += 1
                            elif top1_challenge_ids[idx] == 'regular':
                                correct_count_top1_regular_25 += 1
                            elif top1_challenge_ids[idx] == 'confusing':
                                correct_count_top1_confusing_25 += 1

                            if top1_prompt_ids[idx] == 'discriminative':
                                correct_count_top1_discriminative_25 += 1
                            elif top1_prompt_ids[idx] == 'spacial':
                                correct_count_top1_spacial_25 += 1
                            elif top1_prompt_ids[idx] == 'comprehensive':
                                correct_count_top1_comprehensive_25 += 1
                            elif top1_prompt_ids[idx] == 'fuzzy':
                                correct_count_top1_fuzzy_25 += 1
                            correct_count_top1_25 += 1

                    if max_pro_idx[idx].item() in top1_target_id_50[idx]:
                        if top1_scene_ids[idx] == top1_text_scene_ids[idx]:
                            if top1_challenge_ids[idx] == 'conspicuous':
                                correct_count_top1_conspicuous_50 += 1
                            elif top1_challenge_ids[idx] == 'regular':
                                correct_count_top1_regular_50 += 1
                            elif top1_challenge_ids[idx] == 'confusing':
                                correct_count_top1_confusing_50 += 1

                            if top1_prompt_ids[idx] == 'discriminative':
                                correct_count_top1_discriminative_50 += 1
                            elif top1_prompt_ids[idx] == 'spacial':
                                correct_count_top1_spacial_50 += 1
                            elif top1_prompt_ids[idx] == 'comprehensive':
                                correct_count_top1_comprehensive_50 += 1
                            elif top1_prompt_ids[idx] == 'fuzzy':
                                correct_count_top1_fuzzy_50 += 1
                            correct_count_top1_50 += 1
                    total_count_top1 += 1

                    if top1_challenge_ids[idx] == 'conspicuous':
                        total_count_top1_conspicuous += 1
                    elif top1_challenge_ids[idx] == 'regular':
                        total_count_top1_regular += 1
                    elif top1_challenge_ids[idx] == 'confusing':
                        total_count_top1_confusing += 1
                    
                    if top1_prompt_ids[idx] == 'discriminative':
                        total_count_top1_discriminative += 1
                    elif top1_prompt_ids[idx] == 'spacial':
                        total_count_top1_spacial += 1
                    elif top1_prompt_ids[idx] == 'comprehensive':
                        total_count_top1_comprehensive += 1
                    elif top1_prompt_ids[idx] == 'fuzzy':
                        total_count_top1_fuzzy += 1

        accuracy_top1 = correct_count_top1_25 / total_count_top1
        accuracy_top1_ = correct_count_top1_50 / total_count_top1

        print(f"Top 1 - @25 Correct: {correct_count_top1_25}/{total_count_top1}, Accuracy: {accuracy_top1 * 100:.2f}%")
        print(f"Top 1 - @50 Correct: {correct_count_top1_50}/{total_count_top1}, Accuracy: {accuracy_top1_ * 100:.2f}%")

        print(f"Top 1 - @25 conspicuous:  {correct_count_top1_conspicuous_25}/{total_count_top1_conspicuous}, Accuracy: {(correct_count_top1_conspicuous_25/total_count_top1_conspicuous) * 100:.2f}%")
        print(f"Top 1 - @25 regular:  {correct_count_top1_regular_25}/{total_count_top1_regular}, Accuracy: {(correct_count_top1_regular_25/total_count_top1_regular) * 100:.2f}%")
        print(f"Top 1 - @25 confusing:  {correct_count_top1_confusing_25}/{total_count_top1_confusing}, Accuracy: {(correct_count_top1_confusing_25/total_count_top1_confusing) * 100:.2f}%")

        print(f"Top 1 - @50 conspicuous:  {correct_count_top1_conspicuous_50}/{total_count_top1_conspicuous}, Accuracy: {(correct_count_top1_conspicuous_50/total_count_top1_conspicuous) * 100:.2f}%")
        print(f"Top 1 - @50 regular:  {correct_count_top1_regular_50}/{total_count_top1_regular}, Accuracy: {(correct_count_top1_regular_50/total_count_top1_regular) * 100:.2f}%")
        print(f"Top 1 - @50 confusing:  {correct_count_top1_confusing_50}/{total_count_top1_confusing}, Accuracy: {(correct_count_top1_confusing_50/total_count_top1_confusing) * 100:.2f}%")
       
        print(f"Top 1 - @25 discriminative:  {correct_count_top1_discriminative_25}/{total_count_top1_discriminative}, Accuracy: {(correct_count_top1_discriminative_25/total_count_top1_discriminative) * 100:.2f}%")
        print(f"Top 1 - @25 spacial:  {correct_count_top1_spacial_25}/{total_count_top1_spacial}, Accuracy: {(correct_count_top1_spacial_25/total_count_top1_spacial) * 100:.2f}%")
        print(f"Top 1 - @25 comprehensive:  {correct_count_top1_comprehensive_25}/{total_count_top1_comprehensive}, Accuracy: {(correct_count_top1_comprehensive_25/total_count_top1_comprehensive) * 100:.2f}%")
        print(f"Top 1 - @25 fuzzy:  {correct_count_top1_fuzzy_25}/{total_count_top1_fuzzy}, Accuracy: {(correct_count_top1_fuzzy_25/total_count_top1_fuzzy) * 100:.2f}%")

        print(f"Top 1 - @50 discriminative:  {correct_count_top1_discriminative_50}/{total_count_top1_discriminative}, Accuracy: {(correct_count_top1_discriminative_50/total_count_top1_discriminative) * 100:.2f}%")
        print(f"Top 1 - @50 spacial:  {correct_count_top1_spacial_50}/{total_count_top1_spacial}, Accuracy: {(correct_count_top1_spacial_50/total_count_top1_spacial) * 100:.2f}%")
        print(f"Top 1 - @50 comprehensive:  {correct_count_top1_comprehensive_50}/{total_count_top1_comprehensive}, Accuracy: {(correct_count_top1_comprehensive_50/total_count_top1_comprehensive) * 100:.2f}%")
        print(f"Top 1 - @50 fuzzy:  {correct_count_top1_fuzzy_50}/{total_count_top1_fuzzy}, Accuracy: {(correct_count_top1_fuzzy_50/total_count_top1_fuzzy) * 100:.2f}%")
    
class ModelLossMixin(object):
    def get_refer_loss(self, data_dict):
        
        total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, matching_loss, ceb_loss = self.refer_loss(data_dict['txt_cls_logits'], data_dict['obj_cls_post_logits'], data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['og3d_logits'], 
                                  data_dict['tgt_object_label'], data_dict['tgt_object_id'], data_dict['obj_labels'], data_dict['obj_masks'], data_dict['aggr_point_feat_1'],  data_dict['aggr_text_feat_1'],data_dict['aggr_point_fuse_feature_1'],data_dict['aggr_text_fuse_feature_1'])
        data_dict['total_loss'] = total_loss
        data_dict['og3d_loss'] = og3d_loss
        data_dict['txt_cls_loss'] = txt_cls_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss

        data_dict['matching_loss'] = matching_loss
        data_dict['ceb_loss'] = ceb_loss
        return data_dict
    
    def get_qa_loss(self, data_dict):
        total_loss, og3d_loss, txt_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss = self.qa_loss(data_dict['txt_cls_logits'], data_dict['obj_cls_post_logits'], data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['og3d_logits'], data_dict['answer_scores'],
                                  data_dict['tgt_object_label'], data_dict['tgt_object_id'], data_dict['obj_labels'], data_dict['obj_masks'], data_dict['answer_label'])
        data_dict['total_loss'] = total_loss
        data_dict['og3d_loss'] = og3d_loss
        data_dict['txt_cls_loss'] = txt_cls_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        data_dict['answer_loss'] = answer_loss
        return data_dict

    def get_sqa_loss(self, data_dict):
        total_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss, answer_loss = self.qa_loss(data_dict['obj_cls_post_logits'], data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['answer_scores'],
                                   data_dict['obj_labels'], data_dict['obj_masks'], data_dict['answer_label'])
        data_dict['total_loss'] = total_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        data_dict['answer_loss'] = answer_loss
        return data_dict
        
    def get_caption_loss(self, data_dict):
        total_loss, caption_cls_loss, obj_cls_raw_loss, obj_cls_pre_loss, obj_cls_post_loss = self.caption_loss(data_dict['txt_caption_cls_logit'], data_dict['masked_lm_labels'], data_dict['obj_cls_post_logits'], 
                           data_dict['obj_cls_pre_logits'], data_dict['obj_cls_raw_logits'], data_dict['obj_labels'], data_dict['obj_masks'])
        data_dict['total_loss'] = total_loss
        data_dict['caption_cls_loss'] = caption_cls_loss
        data_dict['obj_cls_raw_loss'] = obj_cls_raw_loss
        data_dict['obj_cls_pre_loss'] = obj_cls_pre_loss
        data_dict['obj_cls_post_loss'] = obj_cls_post_loss
        return data_dict
    
class ModelMetricMixin(object):
    def get_scanrefer_metrics(self, data_dict):
        data_dict['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=1) == data_dict['tgt_object_id'].squeeze(1)).sum().item() / float(len(data_dict['tgt_object_id']))
        # get og acc iou 25 and 50
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
        iou25_correct = 0
        iou50_correct = 0
        iou25_unique_correct = 0
        iou50_unique_correct = 0
        iou25_multiple_correct = 0
        iou50_multiple_correct = 0
        unique_count = 1e-10
        multiple_count = 1e-10
        for i in range(len(og_pred)):
            if data_dict['is_multiple'][i]:
                multiple_count += 1
            else:
                unique_count += 1
            if data_dict['tgt_object_id_iou25'][i, og_pred[i]]:
                iou25_correct += 1
                if data_dict['is_multiple'][i]:
                    iou25_multiple_correct += 1
                else:
                    iou25_unique_correct += 1
            if data_dict['tgt_object_id_iou50'][i, og_pred[i]]:
                iou50_correct += 1
                if data_dict['is_multiple'][i]:
                    iou50_multiple_correct += 1
                else:
                    iou50_unique_correct += 1
        data_dict['og_acc_iou25'] = iou25_correct / float(len(og_pred))
        data_dict['og_acc_iou50'] = iou50_correct / float(len(og_pred))
        data_dict['og_acc_iou25_unique'] = iou25_unique_correct / unique_count
        data_dict['og_acc_iou50_unique'] = iou50_unique_correct / unique_count
        data_dict['og_acc_iou25_multiple'] = iou25_multiple_correct / multiple_count
        data_dict['og_acc_iou50_multiple'] = iou50_multiple_correct / multiple_count
        data_dict['unique_count'] = unique_count
        data_dict['multiple_count'] = multiple_count
        
        # get other
        data_dict['txt_acc'] = torch.sum(torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).item() / float(len(data_dict['tgt_object_label']))
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['og_acc']
        return data_dict

    def get_cross_scene_metrics(self, data_dict):
        data_dict['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=1) == data_dict['tgt_object_id'].squeeze(1)).sum().item() / float(len(data_dict['tgt_object_id']))
        # get og acc iou 25 and 50
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
        iou25_correct = 0
        iou50_correct = 0
        iou25_conspicuous_correct = 0
        iou50_conspicuous_correct = 0
        
        iou25_regular_correct = 0
        iou50_regular_correct = 0
        iou25_confusing_correct = 0
        iou50_confusing_correct = 0

        conspicuous_count = 1e-10
        regular_count = 1e-10
        confusing_count = 1e-10

              
        for i in range(len(og_pred)):
            if data_dict['challenge'][i] == 'conspicuous':
                conspicuous_count += 1
            elif data_dict['challenge'][i] == 'regular':
                regular_count += 1
            elif data_dict['challenge'][i] == 'confusing':
                confusing_count += 1

            if data_dict['tgt_object_id_iou25'][i, og_pred[i]]:
                iou25_correct += 1
                if data_dict['challenge'][i] == 'conspicuous':
                    iou25_conspicuous_correct += 1
                elif data_dict['challenge'][i] == 'regular':
                    iou25_regular_correct += 1
                elif data_dict['challenge'][i] == 'confusing':
                    iou25_confusing_correct += 1

            if data_dict['tgt_object_id_iou50'][i, og_pred[i]]:
                iou50_correct += 1
                if data_dict['challenge'][i] == 'conspicuous':
                    iou50_conspicuous_correct += 1
                elif data_dict['challenge'][i] == 'regular':
                    iou50_regular_correct += 1
                elif data_dict['challenge'][i] == 'confusing':
                    iou50_confusing_correct += 1

        data_dict['og_acc_iou25'] = iou25_correct / float(len(og_pred))
        data_dict['og_acc_iou50'] = iou50_correct / float(len(og_pred))

        data_dict['og_acc_iou25_conspicuous'] = iou25_conspicuous_correct / conspicuous_count
        data_dict['og_acc_iou50_conspicuous'] = iou50_conspicuous_correct / conspicuous_count

        data_dict['og_acc_iou25_regular'] = iou25_regular_correct / regular_count
        data_dict['og_acc_iou50_regular'] = iou50_regular_correct / regular_count

        data_dict['og_acc_iou25_confusing'] = iou25_confusing_correct / confusing_count
        data_dict['og_acc_iou50_confusing'] = iou50_confusing_correct / confusing_count

        data_dict['conspicuous_count'] = conspicuous_count
        data_dict['regular_count'] = regular_count
        data_dict['confusing_count'] = confusing_count

        # get other
        data_dict['txt_acc'] = torch.sum(torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).item() / float(len(data_dict['tgt_object_label']))
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['og_acc']
        return data_dict

    def get_referit3d_metrics(self, data_dict):
        data_dict['og_acc'] = (torch.argmax(data_dict['og3d_logits'], dim=1) == data_dict['tgt_object_id'].squeeze(1)).sum().item() / float(len(data_dict['tgt_object_id']))
        # get og acc iou 25 and 50
        og_pred = torch.argmax(data_dict['og3d_logits'], dim=1)
        easy_correct = 0
        hard_correct = 0
        view_dep_correct = 0
        view_indep_correct = 0
        easy_count = 1e-10
        hard_count = 1e-10
        view_dep_count = 1e-10
        view_indep_count = 1e-10
        for i in range(len(og_pred)):
            if data_dict['is_hard'][i]:
                hard_count += 1
            else:
                easy_count += 1
            if data_dict['is_view_dependent'][i]:
                view_dep_count += 1
            else:
                view_indep_count += 1
            if data_dict['tgt_object_id'][i] == og_pred[i]:
                if data_dict['is_hard'][i]:
                    hard_correct += 1
                else:
                    easy_correct += 1
                if data_dict['is_view_dependent'][i]:
                    view_dep_correct += 1
                else:
                    view_indep_correct += 1
        data_dict['og_acc_easy'] =  easy_correct / easy_count
        data_dict['og_acc_hard'] =  hard_correct / hard_count
        data_dict['og_acc_view_dep'] =  view_dep_correct / view_dep_count
        data_dict['og_acc_view_indep'] =  view_indep_correct / view_indep_count
        data_dict['easy_count'] = easy_count
        data_dict['hard_count'] = hard_count
        data_dict['view_dep_count'] = view_dep_count
        data_dict['view_indep_count'] = view_indep_count
        # get other
        data_dict['txt_acc'] = torch.sum(torch.argmax(data_dict['txt_cls_logits'], dim=1) == data_dict["tgt_object_label"].squeeze(1)).item() / float(len(data_dict['tgt_object_label']))
        data_dict['obj_cls_post_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_pre_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)
        data_dict['obj_cls_raw_acc'] = torch.sum(torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item() + 1e-10)

        data_dict['target_metric'] = data_dict['og_acc']
        return data_dict
        
   