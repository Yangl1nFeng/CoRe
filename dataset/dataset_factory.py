import os
from re import L

from dataset.data_wrapper import *
from dataset.scanrefer import *
from dataset.referit3d import *
from dataset.cross_scene_data import *
from pipeline.registry import registry

@registry.register_dataset("scanrefer")
def get_scanrefer_dataset(split='train', **args):
    dataset = ScanReferDataset(split=split, **args)    
    return dataset

@registry.register_dataset("scanrefer_task")
def get_scanrefer_task_dataset(split='train', tokenizer=None, txt_seq_length=50, pc_seq_length=80, **args):
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = ScanReferDataset(split=split, max_obj_len=pc_seq_length, **args)
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)

@registry.register_dataset("cross_scene_task")
def get_my_data_task_dataset(split='train', tokenizer=None, txt_seq_length=100, pc_seq_length=80, **args):
    print(" Use our get_cross_scene_data_task_dataset function!")
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = Cross_Scene_Dataset(split=split, max_obj_len=pc_seq_length, **args)
    print("txt_seq_length is " + str(txt_seq_length))
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)

@registry.register_dataset("referit3d")
def get_referit3d_dataset(split='train', **args):
    dataset = Referit3DDataset(split=split, **args)    
    return dataset

@registry.register_dataset("referit3d_task")
def get_referit3d_task_dataset(split='train', tokenizer=None, txt_seq_length=50, pc_seq_length=80, **args):
    tokenizer = registry.get_language_model(tokenizer)()
    dataset = Referit3DDataset(split=split, max_obj_len=pc_seq_length, **args)
    return ScanFamilyDatasetWrapper(dataset=dataset, tokenizer=tokenizer, max_seq_length=txt_seq_length, max_obj_len=pc_seq_length)
    
if __name__ == '__main__':
    #dataset = get_scanqa_dataset()
    pass
    
