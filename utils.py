import os
from os.path import exists, join
import json

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data
"""
def get_data_path(mode, encoder):
    paths = {}
    if mode == 'train':
        paths['train'] = '../MatchSum/data/train_multinews_' + encoder + '.jsonl'
        paths['val']   = '../MatchSum/data/val_multinews_' + encoder + '.jsonl'
    else:
        paths['test']  = '../MatchSum/data/test_multinews_' + encoder + '.jsonl'
    return paths
"""
def get_data_path(mode, encoder):
    paths = {}
    if mode == 'train':
        paths['train'] = '../MatchSum/data/train_wikihow_' + encoder + '.jsonl'
        paths['val']   = '../MatchSum/data/val_wikihow_' + encoder + '.jsonl'
    else:
        paths['test']  = '../MatchSum/data/test_wikihow_' + encoder + '.jsonl'
    return paths

def get_result_path(save_path, cur_model):
    result_path = join(save_path, '../result')
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'dec')
    ref_path = join(model_path, 'ref')
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path
