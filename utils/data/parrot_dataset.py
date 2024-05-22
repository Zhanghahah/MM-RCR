# This file is adapted from https://github.com/Zhanghahah/VQA-LLM
# This dataset is from lsvq

import os
import random
# import decord
# import cv2
import json
from PIL import Image
from .gqa_dataset import GQADataset
import utils.data.DST as DST
from utils.utils import get_rank, print_rank_0
from .utils import save_debug_image, save_debug_text
import pickle

import numpy as np
import torch


def load_instruction_from_parrot(data_path):
    """input data: Parrot_USPTO_condition_68w.csv"""
    pass


def parse_pickle(file_path):
    with open(file_path, 'rb') as f:
        ret = pickle.load(f)
    return ret


class ParrotDataset(GQADataset):
    def __init__(self, data_path,
                 data_debug_path,
                 per_sample_graph,
                 tokenizer,
                 graph_processor,
                 add_eos=True, ignore_instruction=True, save_video_feat=False, **kwargs):
        """data_path: data1/zhangyu/yuruijie/"""
        graph_root = f'{data_path}/Parrot_save_files/given-model-processed-info'
        assert os.path.isdir(graph_root), (f"Parrot {graph_root} not found, "
                                           f"you need to check the graph embedding path")

        ann_paths = [
            "Parrot_instruction_files/parrot_instruction_condition_train.json"]  # "LSVQ/LSVQ/a_split_metadata/LSVQ_whole_train_ds_score.json"
        # q_mos_path = os.path.join(data_path, 'a_prompt/prompt_list_noTask.json')
        # q_ass_path = os.path.join(data_path, 'a_prompt/prompt_list_noTask_ass.json')
        #
        # self.catalyst_prompt =  open(
        #     os.path.join(
        #         self.,
        #         "catalyst_prediction_prompt_template.txt"
        #     ), "r").readlines()
        # self.Q_ASS = json.load(open(q_ass_path, 'r'))

        real_ann_paths = []
        for ann_path in ann_paths:
            ann_path = f"{data_path}/{ann_path}"
            real_ann_paths.append(ann_path)
            assert os.path.isfile(
                ann_path), (f"ParrotDataset annotation file {ann_path} not found, "
                            f"you need to identify it from your folder")
        super().__init__(data_path, data_debug_path, per_sample_graph, tokenizer, graph_processor,
                         graph_root, real_ann_paths, annotation_key=None, **kwargs)

    def process_text(self, ann, data_debug_path=None, data_debug_counter=0, first_message=True):
        # random select a question
        ann_dict = json.loads(ann)
        question = ann_dict["instruction"]
        answer = ann_dict["output"]

        instruction = self.prompter(question, with_image=True, first_message=first_message)
        save_debug_text([instruction, answer], data_debug_path, data_debug_counter, get_rank())
        return dict(instruction=instruction, answer=answer)

    def process_graph(self, ann, data_debug_path=None, data_debug_counter=0):
        ann_dict = json.loads(ann)
        graph_idx = ann_dict["graph_id"]
        graph_path = os.path.join(self.graph_root, f"info-{graph_idx}.pkl")
        graph_info = parse_pickle(graph_path)
        graph_feat = torch.tensor(graph_info[3]).unsqueeze(0)
        return graph_feat
