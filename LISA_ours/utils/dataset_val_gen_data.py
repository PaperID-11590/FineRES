import glob
import os
from queue import Empty
import random
import json


from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import transformers

from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset

from .vqa_dataset import VQADataset

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

from model.llava import conversation as conversation_lib
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from datetime import datetime





# _dict_file_path = "/data_16T/tc/huliwen/MMR-main/sencond_time/val_object_part_dict.json"  # 默认字典文件路径

_dict_file_path = "/data_16T/tc/huliwen/MMR-main/sencond_time/val_object_part_dict.json"
_part_whole_dict = None
def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    image_origin_list=[]
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    text_list = []
    sent_idlist = []
    ans_list = []
    for (
        image_path,
        images,
        image_origin,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
        texts,
        sent_id
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        image_origin_list.append(image_origin)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        text_list.append(texts)
        sent_idlist.append(sent_id)

    def load_part_whole_dict(dict_file_path=None):
        """加载part/whole字典文件"""
        global _part_whole_dict, _dict_file_path
        
        if dict_file_path:
            _dict_file_path = dict_file_path
        
        if _part_whole_dict is None:
            if Path(_dict_file_path).exists():
                try:
                    with open(_dict_file_path, "r", encoding="utf-8") as f:
                        _part_whole_dict = json.load(f)
                    print(f"✅ 已加载part/whole字典: {_dict_file_path}")
                    print(f"   包含 {len(_part_whole_dict)} 个条目")
                except Exception as e:
                    print(f"❌ 加载字典文件失败: {e}")
                    _part_whole_dict = {}
            else:
                print(f"⚠️  字典文件不存在: {_dict_file_path}")
                _part_whole_dict = {}
        
        return _part_whole_dict
    def get_part_whole_from_dict(sent_id_or_text, part_whole_dict=None):
        """
        从字典中获取part和whole描述
        
        Args:
            sent_id_or_text: sent_id（整数）或原始文本（字符串）
            part_whole_dict: 可选的字典数据，如果不提供则使用全局加载的字典
        
        Returns:
            tuple: (part_text, whole_text)
        """
        if part_whole_dict is None:
            part_whole_dict = load_part_whole_dict()
        
        result = None
        
        # 如果输入是sent_id（整数）
        if isinstance(sent_id_or_text, (int, str)) and str(sent_id_or_text).isdigit():
            sent_id_str = str(sent_id_or_text)
            result = part_whole_dict.get(sent_id_str)
        
        # 如果输入是文本，则通过original_sentence匹配
        elif isinstance(sent_id_or_text, str):
            result = part_whole_dict.get(sent_id_or_text)
            
        
        if result is None:
            print(f"⚠️  未找到匹配项: {sent_id_or_text}")
            # 如果找不到，返回原文本作为whole，part为空
            fallback_text = str(sent_id_or_text) if not isinstance(sent_id_or_text, str) else sent_id_or_text
            return "", fallback_text
        
        # 根据ans字段决定返回值
        ans_list.append(result.get("ans", "no"))
        if result.get("ans", "no") == "yes":
            part_text = result.get("part", "")
            whole_text = result.get("whole", "")
            if whole_text == "":
                whole_text = result.get("part", "")
        else:
            # ans为"no"时，part_text和whole_text都使用whole
            
            part_text = result.get("whole", "")
            whole_text = result.get("whole", "")
        
        return part_text, whole_text

    def create_part_whole_conversations(text_list, sent_ids_list=None, multiseg_inference=False, dict_file_path=None):
        """
        基于输入的text_list生成part和whole的conversations
        
        Args:
            text_list: 文本列表
            sent_ids_list: 对应的sent_id列表（可选）
            multiseg_inference: 是否为多分割推理
            dict_file_path: 字典文件路径（可选）
        """
        part_conversation_list = []
        whole_conversation_list = []
        
        # 检查输入是否为None或空
        if text_list is None or len(text_list) == 0:
            return part_conversation_list, whole_conversation_list
        
        # 加载字典
        if dict_file_path:
            load_part_whole_dict(dict_file_path)
        part_whole_dict = load_part_whole_dict()
        

        # print("sent_idlist:",sent_idlist)
        sent_idlist1 = sent_idlist[0]
        # 遍历每个text
        texts_to_process = text_list[0] if isinstance(text_list[0], list) else text_list
        # print("texts_to_process:",texts_to_process)
        for i, text in enumerate(texts_to_process):
            if text is None or text.strip() == "":
                continue
            # print("text:???",text)
            # 优先使用sent_id查找，否则使用文本查找
            lookup_key = None
            if sent_idlist1 and i < len(sent_idlist1) and sent_idlist1[i] is not None:
                lookup_key = sent_idlist1[i]
            else:
                lookup_key = text
            
            # 从字典获取part和whole描述
            # print("sent_id:",lookup_key)
            part_text, whole_text = get_part_whole_from_dict(lookup_key, part_whole_dict)
            
            # print(f"文本: '{text}' -> part: '{part_text}', whole: '{whole_text}'")
            
            # 生成part conversation
            if part_text.strip():  # 只有当part_text不为空时才生成part conversation
                conv = conversation_lib.default_conversation.copy()
                conv.messages = []
                
                if part_text == whole_text:
                    # 当part和whole相同时（ans="no"的情况），使用whole的问法
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} in this image? Please output segmentation mask.".format(
                            part_text
                        ),
                    )
                else:
                    # 当part和whole不同时（ans="yes"的情况），使用part的问法
                    conv.append_message(
                        conv.roles[0],
                        DEFAULT_IMAGE_TOKEN
                        + "\n What is {} of the {} in this image? Please output segmentation mask.".format(
                            part_text, whole_text
                        ),
                    )
                
                if multiseg_inference:
                    _seg = "[SEG]"
                    answer = [_seg] * len([part_text])
                    answer = ', '.join(answer[:-1]) + ' and ' + answer[-1] + '.' if len(answer) > 1 else answer[0]
                    conv.append_message(conv.roles[1], answer)
                else:
                    conv.append_message(conv.roles[1], "[SEG].")
                
                part_conversation_list.append(conv.get_prompt())
            
            # 生成whole conversation
            if whole_text.strip():  # 只有当whole_text不为空时才生成whole conversation
                conv = conversation_lib.default_conversation.copy()
                conv.messages = []
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        whole_text
                    ),
                )
                
                if multiseg_inference:
                    _seg = "[SEG]"
                    answer = [_seg] * len([whole_text])
                    answer = ', '.join(answer[:-1]) + ' and ' + answer[-1] + '.' if len(answer) > 1 else answer[0]
                    conv.append_message(conv.roles[1], answer)
                else:
                    conv.append_message(conv.roles[1], "[SEG].")
                
                whole_conversation_list.append(conv.get_prompt())
        
        return part_conversation_list, whole_conversation_list


    part_conversation_list, whole_conversation_list = create_part_whole_conversations(
        text_list
    )

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        # 处理part conversations
        for i in range(len(part_conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            part_conversation_list[i] = part_conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
            
        # 处理whole conversations
        for i in range(len(whole_conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            whole_conversation_list[i] = whole_conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Part conversations
    part_input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in part_conversation_list
    ]
    part_input_ids = torch.nn.utils.rnn.pad_sequence(
        part_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    part_attention_masks = part_input_ids.ne(tokenizer.pad_token_id)

    # Whole conversations
    whole_input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in whole_conversation_list
    ]
    whole_input_ids = torch.nn.utils.rnn.pad_sequence(
        whole_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    whole_attention_masks = whole_input_ids.ne(tokenizer.pad_token_id)


    conv = conversation_lib.conv_templates['chatml'].copy() if conv_type == "chatml" else conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    part_targets = part_input_ids.clone()
    whole_targets = whole_input_ids.clone()

    if conv_type == "llava_v1" or "chatml":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    
    
    for conversation, target in zip(conversation_list, targets):
        # print("conversation:",conversation)
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.sep2 not in conversation:
            break
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if conv_type == "chatml":
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(rou+sep, tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(rou+sep).input_ids) - 2

                if i == 0:
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                # cur_len += round_len
                    
            else:
                parts = rou.split(sep)
                # if len(parts) != 2:
                #     break
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
        if conv_type == "chatml":
            cur_len = total_len
        target[cur_len:] = IGNORE_INDEX

    # 处理part targets
    for conversation, target in zip(part_conversation_list, part_targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.sep2 not in conversation:
            break
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if conv_type == "chatml":
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(rou+sep, tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(rou+sep).input_ids) - 2

                if i == 0:
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                    
            else:
                parts = rou.split(sep)
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
                
        if conv_type == "chatml":
            cur_len = total_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    # 处理whole targets
    for conversation, target in zip(whole_conversation_list, whole_targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if conv.sep2 not in conversation:
            break
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if conv_type == "chatml":
                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(rou+sep, tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(rou+sep).input_ids) - 2

                if i == 0:
                    target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                    
            else:
                parts = rou.split(sep)
                assert len(parts) == 2, (len(parts), rou)
                parts[0] += sep

                if DEFAULT_IMAGE_TOKEN in conversation:
                    round_len = len(tokenizer_image_token(rou, tokenizer))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
                
        if conv_type == "chatml":
            cur_len = total_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    # Truncate if needed
    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            
        if part_input_ids.shape[1] > truncate_len:
            part_input_ids = part_input_ids[:, :truncate_len]
            part_targets = part_targets[:, :truncate_len]
            part_attention_masks = part_attention_masks[:, :truncate_len]
            
        if whole_input_ids.shape[1] > truncate_len:
            whole_input_ids = whole_input_ids[:, :truncate_len]
            whole_targets = whole_targets[:, :truncate_len]
            whole_attention_masks = whole_attention_masks[:, :truncate_len]


        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )
        # print(tokenizer.model_max_length, cur_len, total_len)
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "text_list":text_list[0],
        "ans_list": ans_list,
        "sent_idlist": sent_idlist[0], # 新加内容
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "image_origins": np.stack(image_origin_list,axis=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        
        # 原始数据
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        
        # Part相关数据
        "part_input_ids": part_input_ids,
        "part_labels": part_targets,
        "part_attention_masks": part_attention_masks,
        "part_conversation_list": part_conversation_list,
        
        # Whole相关数据
        "whole_input_ids": whole_input_ids,
        "whole_labels": whole_targets,
        "whole_attention_masks": whole_attention_masks,
        "whole_conversation_list": whole_conversation_list,
        
        # 其他原始字段
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }

class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        num_classes_per_question=1,
        use_expand_question_list=False,
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        local_rank=1,

    ):
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.local_rank = local_rank

        self.datasets = dataset.split("||")
        self.all_datasets = []
        # for dataset in self.datasets:
        #     if dataset == "sem_seg":
        #         self.all_datasets.append(
        #             SemSegDataset(
        #                 base_image_dir,
        #                 tokenizer,
        #                 vision_tower,
        #                 samples_per_epoch,
        #                 precision,
        #                 image_size,
        #                 num_classes_per_sample,
        #                 sem_seg_data,
        #                 num_classes_per_question,
        #                 use_expand_question_list,
        #                 local_rank
        #             )
        #         )
        #     elif dataset == "refer_seg":
        #         self.all_datasets.append(
        #             ReferSegDataset(
        #                 base_image_dir,
        #                 tokenizer,
        #                 vision_tower,
        #                 samples_per_epoch,
        #                 precision,
        #                 image_size,
        #                 num_classes_per_sample,
        #                 refer_seg_data,
        #                 num_classes_per_question,
        #                 use_expand_question_list,
    
        #             )
        #         )
        #     elif dataset == "vqa":
        #         self.all_datasets.append(
        #             VQADataset(
        #                 base_image_dir,
        #                 tokenizer,
        #                 vision_tower,
        #                 samples_per_epoch,
        #                 precision,
        #                 image_size,
        #                 vqa_data,
        #             )
        #         )
        #     elif dataset == "reason_seg":
        #         self.all_datasets.append(
        #             ReasonSegDataset(
        #                 base_image_dir,
        #                 tokenizer,
        #                 vision_tower,
        #                 samples_per_epoch,
        #                 precision,
        #                 image_size,
        #                 num_classes_per_sample,
        #                 reason_seg_data,
        #                 explanatory,
        #             )
        #         )
        #     elif dataset == "multi_part_reason_seg":
        #         self.all_datasets.append(
        #             MultiPartReasonSegDataset(
        #                 base_image_dir,
        #                 tokenizer,
        #                 vision_tower,
        #                 samples_per_epoch,
        #                 precision,
        #                 image_size,
        #                 num_classes_per_sample,
        #                 use_expand_question_list
        #             )
        #         )
                

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference




class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
        json_name=""

    ):
        self.base_image_dir = base_image_dir
        self.multiseg_inference = False
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            if ds == "MultiPartReasonSeg":
                json_file_name = os.path.join(self.base_image_dir, "MMR", json_name)
                with open(json_file_name, 'r') as f:
                    reason_file = json.load(f)
                self.reason_seg_data = reason_file
                self.data_type = 'multi_part_reason_seg'
            elif ds == "RefCOCOm":
                json_file_name = os.path.join(self.base_image_dir, 'refer_seg/RefCOCOm/annotations', json_name)
                with open(json_file_name, 'r') as f:
                    refer_file = json.load(f)
                self.refer_file = refer_file
                self.data_type = "refcocom"    
        
            else:
                images = glob.glob(
                    os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
                )
                self.images = images
                self.data_type = "reason_seg"
                
        elif len(splits) == 3:
            ds, splitBy, split = splits
            if 'multi' in ds:
                self.multiseg_inference = True
                ds = ds.split('multi')[-1]
            refer_api = REFER(os.path.join(self.base_image_dir, 'refer_seg'), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        self.base_image_dir, "refer_seg/images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        self.base_image_dir,
                        "refer_seg/images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size) 
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        elif self.data_type == "multi_part_reason_seg":
            return len(self.reason_seg_data)
        elif self.data_type == "refcocom":
            return len(self.refer_file)
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        
        elif self.data_type == "multi_part_reason_seg":
            image_info = self.reason_seg_data[idx]
            if "file_name" in image_info:    
                image_root = os.path.join(self.base_image_dir, 'refer_seg/images/mscoco/images')
                image_path = os.path.join(image_root, image_info["file_name"])
            anns = image_info['annotations']
            question = image_info['questions'] 
            gt_answer = image_info['answers']
            text_answers = image_info['text_answers']
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = True
            sampled_sents = question
            sampled_answers = text_answers
            
        elif self.data_type == "refcocom":
            image_info = self.refer_file[idx]
            image_root = os.path.join(self.base_image_dir, 'refer_seg/images/mscoco/images/train2014')
            image_path = os.path.join(image_root, image_info['img_name'])
            gt_answer_name = str(image_info['segment_id']) + ".png"
            gt_answer_path = os.path.join(self.base_image_dir, "refer_seg/RefCOCOm/masks", gt_answer_name)
            
            sampled_sents = []
            sampled_sent_ids = []  # 添加sent_id列表
            
            for sent in image_info['sentences']:
                sampled_sents.append(sent['sent'].strip().lower())
                sampled_sent_ids.append(sent.get('sent_id', None))  # 获取sent_id，如果不存在则为None
                
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
            
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        texts = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        _seg = "[SEG]" 
        multi_sample_num = [6, 5, 4]
        multi_sample_index = 0

        while i < len(sampled_sents):
            conv.messages = []
            if self.multiseg_inference:
                sample_num = multi_sample_num[multi_sample_index]
                texts = [sampled_sents[k].strip() for k in range(i, i+sample_num)] if len(sampled_sents) - i >= sample_num else [sampled_sents[k].strip() for k in range(i, len(sampled_sents))]
                text = ', '.join(texts[:-1]) + ' and {}'.format(texts[-1]) if len(texts) > 1 else texts[0]
                
            else:
                if self.data_type == "multi_part_reason_seg":
                    text = sampled_sents[i].strip()
                    _seg = sampled_answers[i].format(seg="[SEG]")
                else:
                    text = sampled_sents[i].strip()
                    _seg = "[SEG]"
            
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "{}.".format(_seg))
                
            else:
                texts.append(text)
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                if self.multiseg_inference:
                    answer = [_seg] * len(texts)
                    answer = ', '.join(answer[:-1]) + ' and ' + answer[-1] + '.' if len(answer) > 1 else answer[0]
                    conv.append_message(conv.roles[1], answer)
                else:
                    conv.append_message(conv.roles[1], "{}.".format(_seg))
            conversations.append(conv.get_prompt())
            if self.multiseg_inference:
                i += sample_num
                multi_sample_index = (multi_sample_index + 1) % len(multi_sample_num)
            else:
                i += 1

        # preprocess image for clip
        imageorigin = image
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
                
        elif self.data_type == "multi_part_reason_seg":
            masks = []
            for answer_list in gt_answer:
                for answer in answer_list:
                    rle = answer["segmentation"]
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)
                    masks.append(m)   
                    
        elif self.data_type == "refcocom":
            masks = []
            gt_mask = cv2.imread(gt_answer_path) # [h, w, c], max_value: 255, min_value: 0
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            gt_mask = gt_mask / 255
            for i in range(len(sampled_sents)):
                masks.append(gt_mask)
        
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        # 根据数据类型返回不同的信息
        if self.data_type == "refcocom":
            # 对于refcocom数据集，返回sent_id信息
            return (
                image_path,
                image,
                imageorigin,
                image_clip,
                conversations,
                masks,
                labels,
                resize,
                None,
                None,
                inference,
                texts,
                sampled_sent_ids  # 添加sent_id列表
            )
        else:
            # 对于其他数据集，保持原有返回格式
            return (
                image_path,
                image,
                imageorigin,
                image_clip,
                conversations,
                masks,
                labels,
                resize,
                None,
                None,
                inference,
                texts
            )

