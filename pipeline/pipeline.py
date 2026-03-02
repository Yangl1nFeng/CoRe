from abc import ABC, abstractmethod
from pipeline.registry import registry
import torch
from torch.optim import AdamW
from pipeline.pipeline_mixin_old import *
from tqdm import tqdm
import numpy as np
from model.vision.basic_modules import generate_causal_mask, Attn_Fusion, ViewWeightMLP, MappingMLP, GPO, AP, RoMa
from torch.optim.lr_scheduler import StepLR

'''
Base class for all pipelines
'''

class Pipeline(ABC):
    @abstractmethod
    def initialize(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def end(self):
        pass

    def run_all(self):
        self.initialize()
        self.run()
        self.end()

class OptimusPrimePipeline(Pipeline, NormalDataloaderMixin, ModelOptimizationMixin, ModelEvaluationMixin, ModelMetricMixin, ModelLossMixin):
    def __init__(self, cfg):
        # build saver and logger
        if not cfg['eval_task']:
            self.logger = registry.get_utils(cfg['logger']['name'])(cfg)
        self.saver = registry.get_utils(cfg['saver']['name'])(**cfg['saver']['args'])
       
        # build model
        self.lang_encoder = registry.get_language_model(cfg['lang_encoder']['name'])(**cfg['lang_encoder']['args']).cuda()
        self.point_encoder = registry.get_vision_model(cfg['point_encoder']['name'])(**cfg['point_encoder']['args']).cuda()
        self.unified_encoder = registry.get_vision_model(cfg['unified_encoder']['name'])(**cfg['unified_encoder']['args']).cuda()
        self.ground_head = registry.get_other_model(cfg['ground_head']['name'])(**cfg['ground_head']['args']).cuda()
        self.point_fusion_head = AP(768).cuda()
        self.text_fusion_head = AP(768).cuda()
        self.view_mlp = ViewWeightMLP(input_dim=768).cuda()
        self.point_mapping = MappingMLP(768).cuda()
        self.text_mapping = MappingMLP(768).cuda()
        self.point_mapping_2 = MappingMLP(768).cuda()
        self.text_mapping_2 = MappingMLP(768).cuda()

        file_path = cfg['point_encoder']['args']['path']
        point_encoder_weights = torch.load(file_path, map_location="cpu")
        self.point_encoder.load_state_dict(point_encoder_weights)
        print("Successfully loaded point_encoder weights.")

        # load task
        self.task = cfg['task']
        self.eval_task = cfg['eval_task']
        
        # build dataset
        if self.task == 'scanrefer' or self.task == 'referit3d' or self.task == 'cross_scene':
            self.train_dataset = registry.get_dataset(cfg['refer_dataset']['name'])(split='train', **cfg['refer_dataset']['args'])
            self.test_dataset = registry.get_dataset(cfg['refer_dataset']['name'])(split='val', **cfg['refer_dataset']['args'])
        else:
            raise NotImplementedError("task " + self.task + " is not implemented")
      
        # load optimize config
        self.batch_size = cfg['batch_size']
        self.learning_rate = cfg['learning_rate']
        self.grad_norm = cfg['grad_norm']
        self.epochs = cfg['epochs']
        self.warmup_steps = cfg['warmup_steps']
        
        # build dataloaer
        self.build_train_test_dataloader()
        
        # add parameters
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters += self.no_decay_param_group(self.lang_encoder.named_parameters(), self.learning_rate * cfg['lang_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(self.point_encoder.named_parameters(), self.learning_rate * cfg['point_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(self.unified_encoder.named_parameters(), self.learning_rate * cfg['unified_lr_mul'])
        optimizer_grouped_parameters += self.no_decay_param_group(self.ground_head.named_parameters(), self.learning_rate)

        optimizer_grouped_parameters += self.no_decay_param_group(self.point_fusion_head.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.text_fusion_head.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.view_mlp.named_parameters(), self.learning_rate)
        
        optimizer_grouped_parameters += self.no_decay_param_group(self.point_mapping.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.text_mapping.named_parameters(), self.learning_rate)

        optimizer_grouped_parameters += self.no_decay_param_group(self.point_mapping_2.named_parameters(), self.learning_rate)
        optimizer_grouped_parameters += self.no_decay_param_group(self.text_mapping_2.named_parameters(), self.learning_rate)
        
        
        # build optimizer
        self.optimizer = AdamW(optimizer_grouped_parameters)
        
        self.parameters = []
        for p in optimizer_grouped_parameters:
            self.parameters.extend(p['params'])


        # build scheduler
        total_steps = self.epochs * len(self.train_data_loader)
        self.total_steps = total_steps
        print("total_steps {}".format(total_steps))

        self.scheduler = StepLR(optimizer=self.optimizer, step_size=0.4*total_steps, gamma=0.1)
       
        # build loss function
        if self.task == 'scanrefer' or self.task == 'referit3d' or self.task == 'cross_scene':
            self.refer_loss = registry.get_optimizer(cfg["refer_loss"]['name'])
        
        # restore model
        if cfg['restore_model']:
            self.restore_model()
        total_params = sum(p.numel() for p in self.parameters if p.requires_grad)
        
    def initialize(self):
        pass
    
    '''
    Hierarchical
    train
        forward
        get loss
        get metric
        record train
    eval
        forward
        get metric
        process metric
        record eval
    '''    
    def run(self):
        best_target_metric = -np.inf # assuming higher is better
        for epoch in range(self.epochs):
            if self.eval_task:
                self.eval(epoch)
                break
            # train
            self.train(epoch)
            if epoch % 10 == 0:
                self.save_model()
                self.eval(epoch)
        self.save_model()
        self.eval(self.epochs)

    def train(self, epoch):
        self.set_model_state('train')
        for i, data_dict in enumerate(tqdm(self.train_data_loader)):
            # add step and total steps to data_dict
            data_dict['cur_step'] = epoch * len(self.train_data_loader) + i 
            data_dict['total_steps'] = self.total_steps
            
            # forward
            data_dict = self.forward_one(data_dict, is_train=True)
        
            # calculate loss
            data_dict = self.get_loss(data_dict)
            
            # calculate metrics
            data_dict = self.get_metrics(data_dict)
            
            # optimize
            loss = data_dict['total_loss']
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, self.grad_norm
            )
            data_dict['grad_norm'] = grad_norm
            self.optimizer.step()
            self.optimizer.zero_grad()     
            self.scheduler.step()
            
            # record
            step = epoch * len(self.train_data_loader) + i 
            self.record_train_step(data_dict, step)
            
    def eval(self, epoch):
        print("start evaluation on test set")
        self.set_model_state('eval')
        # build eval dict
        if self.task == 'scanrefer':
            self.eval_matching_grounding(epoch)
        if self.task == 'cross_scene':
            self.eval_matching_grounding(epoch)
        elif self.task == 'referit3d':
            self.eval_matching_grounding(epoch)
    

    def forward_one(self, data_dict, is_train):
        # prepare data
        self.prepare_data(data_dict)
        
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1

        # ''' mask '''
        spacial = [2392, 2067, 2327, 3953, 2187, 2157, 2415, 2379, 2521, 2682, 2917, 2503, 2648, 5516, 19754, 7471, 9876, 5903, 19581, 4500, 2369, 3875, 2247, 2408, 4218, 2105, 2090, 2104, 2058, 2646, 2114]
        color = [2417, 2630, 2665, 3756, 4589, 6379, 5061, 2829, 2317, 2304, 3897, 3165, 2751, 28799, 28653, 20920, 5572, 22222, 9724, 11034, 2566, 5003, 3673, 11840, 18237, 22088, 18399, 1047, 23867]
        shape = [4799, 3384, 6081, 3221, 13338, 14902, 12538, 4717, 21346, 11841, 3730, 3082, 2422, 18499, 12853, 5931, 5898, 18001, 6953, 10966, 19412, 17517, 8903, 10683, 12923, 12121, 12392, 14692, 9753, 7720, 5318, 5509, 2962, 8782, 3165, 2751, 4421, 27729, 21274, 6557, 18749, 2310, 10966, 9353, 27201, 26189, 18676, 6967, 29071, 26572, 1048, 1055, 1048, 1055, 21987, 10336, 7956, 10806, 9203, 16108, 12052, 23476, 14965, 2461, 8352, 4257, 2235, 2502, 4629, 5744, 6389, 17876, 24750, 14023, 18970, 9242, 2002, 11918, 2004, 8206, 5490, 25147, 12313, 16095, 1054, 4338, 5044, 20864, 22340, 2675]
        objects = [3242, 8397, 2795, 7251, 10682, 10682, 2793, 9705, 5239, 20053, 4624, 4624, 11142, 15475, 2338, 2338, 2341, 4303, 3332, 3645, 5259, 13536, 16641, 16641, 23135, 23135, 2694, 2694, 16716, 16716, 17428, 17428, 18302, 18302, 7752, 23462, 11848, 21674, 7198, 7198, 6457, 23442, 9378, 9378, 18097, 18097, 16247, 16247, 4675, 4675, 7759, 2795, 7759, 7251, 3242, 22936, 3242, 27205, 22936, 27205, 14533, 3698, 14533, 6681, 6457, 11002, 6457, 14694, 11848, 3259, 11848, 4981, 10257, 24213, 3861, 4620, 2396, 2840, 4169, 5265, 6743, 10801, 11446, 14529, 25545, 25545, 3269, 4264, 6546, 4870, 18781, 18781, 2808, 2338, 2808, 2808, 14694, 11002, 28279, 6397, 20452, 20452, 10005, 17860, 10135, 27997, 5894, 21098, 9212, 9212, 10437, 14186, 2422, 4597, 2723, 10437, 2723, 14186, 2723, 2723, 3681, 3332, 3332, 3332, 4853, 3332, 11048, 10958, 10958, 6457, 6457, 11848, 8248, 11848, 22569, 11669, 2064, 11669, 18484, 7752, 5239, 7752, 20053, 18834, 28287, 5108, 23788, 10714, 10714, 4084, 3357, 29372, 29372, 15747, 15747, 11673, 28352, 11134, 25633, 5877, 19963, 2192, 2192, 8829, 8829, 7381, 7381, 2482, 3765, 4316, 4683, 10165, 21066, 9055, 18580, 4049, 6242, 14582, 2064, 12977, 12977, 7390, 6904, 6904, 11868, 11868, 7815, 7815, 25850, 25850, 11868, 11868, 10257, 14513, 10257, 27259, 6865, 6865, 9346, 9346, 16641, 13065, 16641, 22497, 23135, 13065, 23135, 22497, 5239, 2341, 5239, 4303, 17828, 17828, 5259, 4853, 5259, 11048, 3861, 4853, 3861, 11048, 2396, 4853, 2396, 11048, 28279, 8473, 28279, 19485, 11002, 8473, 11002, 19485, 10682, 22936, 10682, 27205, 2793, 10005, 2793, 17860, 20452, 11687, 20452, 19586, 2173, 2173, 2795, 2795, 20619, 20619, 2795, 5479, 2795, 7190, 13541, 14006, 13541, 9111, 13541, 13304, 14006, 14006, 28647, 28647, 28647, 9111, 28647, 13304, 18781, 2007, 4870, 18781, 2007, 4870, 3269, 3269, 5949, 8026, 5949, 8026, 11669, 8026, 11669, 8026, 2858, 7334, 3682, 27864, 6943, 3846, 6710, 25877, 9019, 6269, 4624, 3242, 4624, 8397, 3347, 14708, 3347, 14708, 4157, 2795, 4157, 7251, 2217, 2795, 2217, 7251, 29372, 29372, 14934, 3242, 14934, 8397, 2723, 22936, 2723, 27205, 5894, 5470, 5894, 4599, 19475, 2795, 19475, 7251, 3347, 4675, 3347, 24094, 6847, 19571, 6457, 11002, 8473, 6457, 11002, 19485, 11848, 3259, 9111, 11848, 3259, 13304, 11848, 8248, 9111, 11848, 8248, 13304, 11848, 4951, 11848, 7286, 7198, 13523, 7198, 22281, 6457, 20452, 6457, 20452, 11848, 2835, 11848, 4272, 7198, 6904, 7198, 6904, 6457, 6904, 6457, 6904, 7226, 7226, 5723, 7752, 5723, 23462, 5723, 5259, 5723, 13536, 3108, 1997, 22497, 9378, 9378, 4318, 4318, 3707, 2604, 3707, 7923, 3707, 3707, 14533, 10810, 14533, 25946, 10654, 10654, 7370, 18755, 13788, 13788, 2543, 22885, 22885, 13433, 13433, 11865, 11865, 14068, 4524, 14068, 8641, 14513, 27259, 7997, 10818, 14513, 10818, 27259, 5435, 14513, 5435, 27259, 4511, 14513, 4511, 27259, 5527, 10810, 25946, 9121, 10899, 2377, 2377, 3635, 6847, 3635, 19571, 29449, 29449, 6912, 7997, 6912, 18105, 13272, 13523, 13272, 22281, 12873, 12873, 10516, 3608, 10516, 7395, 27090, 6847, 27090, 19571, 26783, 14513, 26783, 27259, 8362, 4524, 8362, 8641, 4770, 2795, 4770, 7251, 29379, 2795, 29379, 7251, 2795, 5093, 2795, 2795, 5093, 7251, 14957, 14957, 2208, 10122, 2208, 22659, 10877, 3698, 10877, 6681, 2604, 2208, 2604, 2399, 7433, 7433, 2944, 3345, 2944, 4499, 23853, 9121, 2482, 9121, 3765, 2895, 3275, 2895, 4481, 10658, 14421]
        mask_id_1 = torch.tensor(spacial).cuda()
        mask_id_2 = torch.tensor(color + shape + objects).cuda()

        if is_train:
            txt_masks_1 = data_dict['txt_masks']  
            txt_masks_2 = data_dict['txt_masks']
            txt_ids = data_dict['txt_ids'] 
            batch_size, word_num = txt_masks_1.shape
            is_target_id_1 = torch.isin(txt_ids, mask_id_1).float().cuda() 
            is_target_id_2 = torch.isin(txt_ids, mask_id_2).float().cuda() 
            rand_mask = torch.rand_like(txt_masks_1.float()).cuda()  
            mask_condition = (is_target_id_1 == 1) * (rand_mask < 0.1) * (txt_masks_1 == 1)
            txt_masks_1[mask_condition == 1] = 0
            mask_condition = (is_target_id_2 == 1) * (rand_mask < 0.1) * (txt_masks_2 == 1)
            txt_masks_2[mask_condition == 1] = 0
            final_mask = (txt_masks_1 == 1) & (txt_masks_2 == 1)
            txt_masks_1 = final_mask
        else:
            txt_masks_1 = data_dict['txt_masks']
            txt_masks_2 = data_dict['txt_masks']
        
       

        if self.task == 'caption':
            causal_mask = generate_causal_mask(data_dict['txt_ids'].shape[1]).unsqueeze(0).repeat(data_dict['txt_ids'].shape[0], 1, 1).cuda()
            lang_basic_features = self.lang_encoder(data_dict['txt_ids'], causal_mask).last_hidden_state
        else:
            lang_basic_features_1 = self.lang_encoder(data_dict['txt_ids'], txt_masks_1).last_hidden_state
        point_basic_features_1, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(), data_dict['obj_locs'], data_dict['obj_masks'], data_dict['obj_sem_masks'], 
                                                                                        data_dict['obj_labels'], data_dict['cur_step'], data_dict['total_steps'], is_train=is_train)

        indices = torch.arange(data_dict['txt_masks'].size(1), device=data_dict['txt_masks'].device).expand_as(data_dict['txt_masks'])
        masked_indices = torch.where(data_dict['txt_masks'] == 1, indices, torch.tensor(-1, device=data_dict['txt_masks'].device))
        last_indices_text = masked_indices.max(dim=1).values + 1

        aggr_text_feat_1 = self.text_fusion_head(lang_basic_features_1, last_indices_text)

        indices = torch.arange(data_dict['obj_masks'].size(1), device=data_dict['obj_masks'].device).expand_as(data_dict['obj_masks'])
        masked_indices = torch.where(data_dict['obj_masks'] == 1, indices, torch.tensor(-1, device=data_dict['obj_masks'].device))
        last_indices = masked_indices.max(dim=1).values + 1

        aggr_point_feat_1 = self.point_fusion_head(point_basic_features_1, last_indices)

        # unifed language entity transformer
        if self.task == 'caption':
            language_fuse_feature, point_fuse_feature  = self.unified_encoder(lang_basic_features, data_dict['txt_masks'], point_basic_features_1, data_dict['obj_locs'], data_dict['obj_masks'], data_dict['tgt_object_id'], True)
        else:
            language_fuse_feature_1, point_fuse_feature_1  = self.unified_encoder(lang_basic_features_1, txt_masks_1, point_basic_features_1, data_dict['obj_locs'], data_dict['obj_masks'])
        aggr_point_fuse_feature_1 = self.point_fusion_head(point_fuse_feature_1, last_indices)

        aggr_text_fuse_feature_1 = self.text_fusion_head(language_fuse_feature_1, last_indices_text)
        
        # task head
        txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature_1, point_fuse_feature_1, point_features_pre, data_dict['obj_masks'])
        
        data_dict['txt_cls_logits'] = txt_cls_logits
        data_dict['obj_cls_post_logits'] = obj_cls_post_logits
        data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits
        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        data_dict['og3d_logits'] = og3d_logits
    

        data_dict['aggr_point_feat_1'] = aggr_point_feat_1
        data_dict['aggr_text_feat_1'] = aggr_text_feat_1
        data_dict['aggr_point_fuse_feature_1'] = aggr_point_fuse_feature_1
        data_dict['aggr_text_fuse_feature_1'] = aggr_text_fuse_feature_1
        
        return data_dict


    def forward_one_eval(self, data_dict):
        # prepare data
        self.prepare_data(data_dict)

        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1
        # basic feature extracter
        # point_features_pre_spatial is point features before spatial reasonging
        if self.task == 'caption':
            causal_mask = generate_causal_mask(data_dict['txt_ids'].shape[1]).unsqueeze(0).repeat(data_dict['txt_ids'].shape[0], 1, 1).cuda()
            lang_basic_features = self.lang_encoder(data_dict['txt_ids'], causal_mask).last_hidden_state
        else:
            lang_basic_features = self.lang_encoder(data_dict['txt_ids'], data_dict['txt_masks']).last_hidden_state
       
        txt_masks = data_dict['txt_masks']

        point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(data_dict['obj_fts'].float(), data_dict['obj_locs'], data_dict['obj_masks'], data_dict['obj_sem_masks'], 
                                                                                          data_dict['obj_labels'], data_dict['cur_step'], data_dict['total_steps'], is_train=False)
        point_matching_features = self.point_mapping(point_basic_features)
        lang_matching_features = self.text_mapping(lang_basic_features)

        aggr_point_feat = self.point_fusion_head(point_matching_features, data_dict['obj_masks'])
        aggr_text_feat = self.text_fusion_head(lang_matching_features, txt_masks)
        
        # unifed language entity transformer
        if self.task == 'caption':
            language_fuse_feature, point_fuse_feature = self.unified_encoder(lang_basic_features, data_dict['txt_masks'], point_basic_features, data_dict['obj_locs'], data_dict['obj_masks'], data_dict['tgt_object_id'], True)
        else:
            language_fuse_feature, point_fuse_feature = self.unified_encoder(lang_basic_features, txt_masks, point_basic_features, data_dict['obj_locs'], data_dict['obj_masks'])
        
        # task head
        txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature, point_fuse_feature, point_features_pre, data_dict['obj_masks'])
        
        data_dict['txt_cls_logits'] = txt_cls_logits
        data_dict['obj_cls_post_logits'] = obj_cls_post_logits
        data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits
        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        data_dict['og3d_logits'] = og3d_logits
        data_dict['aggr_point_feat'] = aggr_point_feat
        data_dict['aggr_text_feat'] = aggr_text_feat
        
        return data_dict
    
    def get_loss(self, data_dict):
        if self.task == 'scanrefer' or self.task == 'referit3d' or self.task == 'cross_scene':
            data_dict = self.get_refer_loss(data_dict)
        if self.task == 'cross_scene':
            data_dict = self.get_refer_loss(data_dict)
        return data_dict
    
    def get_metrics(self, data_dict):
        if self.task == 'scanrefer':
            data_dict = self.get_scanrefer_metrics(data_dict)
        elif self.task == 'cross_scene':
            data_dict = self.get_cross_scene_metrics(data_dict)
        elif self.task == 'referit3d':
            data_dict = self.get_referit3d_metrics(data_dict)
        return data_dict
     
    def record_train_step(self, data_dict, step):
        log_dict = {
            # basic info
            'step': step,
            'lr': self.scheduler.get_last_lr()[0],
            'grad_norm': data_dict['grad_norm'].item(),
            # shared loss
            'total_loss': data_dict['total_loss'].item(),
            'obj_cls_raw_loss': data_dict['obj_cls_raw_loss'].item(),
            'obj_cls_pre_loss': data_dict['obj_cls_pre_loss'].item(),
            'obj_cls_post_loss': data_dict['obj_cls_post_loss'].item(),
            # shared acc
            'obj_cls_raw_acc': data_dict['obj_cls_raw_acc'],
            'obj_cls_pre_acc': data_dict['obj_cls_pre_acc'],
            'obj_cls_post_acc': data_dict['obj_cls_post_acc'],
        }
        if self.task == 'scanrefer' or self.task == 'referit3d' or self.task == 'cross_scene':
            log_dict.update({
                # loss
                'og3d_loss': data_dict['og3d_loss'].item(),
                'txt_cls_loss': data_dict['txt_cls_loss'].item(),
                # acc
                'og_acc': data_dict['og_acc'],
                'txt_acc': data_dict['txt_acc'],
            })
        for k in list(log_dict.keys()):
            log_dict['train/' + k] = log_dict.pop(k)
        self.logger.log(log_dict, step=step)
            
    def record_eval_step(self, eval_dict, epoch):
        for key in eval_dict.keys():
            if self.eval_task:
                print('test_' + key, eval_dict[key])
            else:
                print('test_' + key, eval_dict[key])
                self.logger.log({'test/' + key: eval_dict[key]}, step = (epoch + 1) * len(self.train_data_loader))
    
    def restore_model(self):
        state_dict = self.saver.restore_dict()
        self.lang_encoder.load_state_dict(state_dict['lang_encoder'])
        self.point_encoder.load_state_dict(state_dict['point_encoder'])
        self.unified_encoder.load_state_dict(state_dict['unified_encoder'])
        self.ground_head.load_state_dict(state_dict['ground_head'])
        try:
            self.point_fusion_head.load_state_dict(state_dict['point_fusion_head'])
            self.text_fusion_head.load_state_dict(state_dict['text_fusion_head'])
            self.view_mlp.load_state_dict(state_dict['view_mlp'])

            self.point_mapping.load_state_dict(state_dict['point_mapping'])
            self.text_mapping.load_state_dict(state_dict['text_mapping'])
            self.point_mapping_2.load_state_dict(state_dict['point_mapping_2'])
            self.text_mapping_2.load_state_dict(state_dict['text_mapping_2'])
        except: 
            print("fail to load our params")
    
    def save_model(self):
        self.saver.save_dict({'lang_encoder': self.lang_encoder.state_dict(),
                              'point_encoder': self.point_encoder.state_dict(),
                              'unified_encoder': self.unified_encoder.state_dict(),
                              'ground_head': self.ground_head.state_dict(),
                              'point_fusion_head': self.point_fusion_head.state_dict(),
                              'text_fusion_head': self.text_fusion_head.state_dict(),
                              'view_mlp': self.view_mlp.state_dict(),
                              'point_mapping': self.point_mapping.state_dict(),
                              'text_mapping': self.text_mapping.state_dict(),
                              'point_mapping_2': self.point_mapping_2.state_dict(),
                              'text_mapping_2': self.text_mapping_2.state_dict()
                              })
    
    def set_model_state(self, state='train'):
        assert state in ['train', 'eval']
        torch.cuda.empty_cache()
        if state == 'train':
            self.lang_encoder.train()
            self.point_encoder.train()
            self.unified_encoder.train()
            self.ground_head.train()
            self.point_fusion_head.train()
            self.text_fusion_head.train()
            self.view_mlp.train()
            self.point_mapping.train()
            self.text_mapping.train()
            self.point_mapping_2.train()
            self.text_mapping_2.train()
        else:
            self.lang_encoder.eval()
            self.point_encoder.eval()
            self.unified_encoder.eval()
            self.ground_head.eval()
            self.point_fusion_head.eval()
            self.text_fusion_head.eval()
            self.view_mlp.eval()
            self.point_mapping.eval()
            self.text_mapping.eval()
            self.point_mapping_2.eval()
            self.text_mapping_2.eval()
                 
    def end(self):
        pass
