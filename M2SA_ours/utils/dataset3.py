import glob
import os
from queue import Empty
import random
import json

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
from .multi_part_reason_seg_dataset import MultiPartReasonSegDataset
import re

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

from model.llava import conversation as conversation_lib
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


# t5_model = T5ForConditionalGeneration.from_pretrained("/data_16T/tc/huliwen/polygon-transformer/test_model/t5_part_detection_3")
# t5_tokenizer = T5Tokenizer.from_pretrained("/data_16T/tc/huliwen/polygon-transformer/dataroot/models/google/flan-t5-large")


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

    # def extract_part_whole_with_t5(sentence): 
    #     """使用T5从referring expression中提取part和whole/object描述"""
    #     if t5_model is None or t5_tokenizer is None:
    #         # Fallback: 返回原始sentence
    #         return sentence, sentence
        
    #     try:
    #         prompt = f"""Does this reference describe a part of an object?
    #         Extract the 'part' and 'whole' (or 'object') from the reference expression below:
    #         Reference: "{sentence}"
            
    #         """
            
    #         input_encoding = t5_tokenizer(
    #             prompt,
    #             return_tensors='pt',
    #             max_length=128,
    #             truncation=True,
    #             padding=True
    #         )
    #         with torch.no_grad():
    #             outputs = t5_model.generate(
    #                 input_ids=input_encoding['input_ids'],
    #                 attention_mask=input_encoding['attention_mask'],
    #                 max_length=128,
    #                 num_beams=4,
    #                 early_stopping=True,
    #                 do_sample=False
    #             )
            
    #         prediction = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         # print("response:", response)

    #         # 解析 yes/no
    #         match_label = re.match(r'^(yes|no)', prediction, re.IGNORECASE)
            

    #         # 提取 part
    #         part_match = re.search(r'part:\[(.*?)\]', prediction)
    #         part = part_match.group(1).strip() if part_match else sentence

    #         # 提取 whole/object
    #         whole_match = re.search(r'whole:\[(.*?)\]', prediction)
    #         object_match = re.search(r'object:\[(.*?)\]', prediction)

    #         if whole_match:
    #             whole = whole_match.group(1).strip()
    #         elif object_match:
    #             whole = object_match.group(1).strip()
    #         else:
    #             whole = sentence

    #         return part, whole
        
    #     except Exception as e:
    #         print("原错误：",sentence)
    #         print(f"Error in T5 part/whole extraction: {e}")
    #         return sentence, sentence

    def create_part_whole_conversations(sampled_classes_list, questions_list):
        """基于提取的part/whole描述重新构建conversations"""
        part_conversation_list = []
        whole_conversation_list = []
        
        # 检查输入是否为None（验证集可能没有这些数据）
        if sampled_classes_list is None or questions_list is None:
            # 对于验证集，直接复制原始conversations
            for _ in conversation_list:
                part_conversation_list.append(_)
                whole_conversation_list.append(_)
            return part_conversation_list, whole_conversation_list
        
        # 遍历每个batch中的sample
        for batch_idx, (sampled_classes, questions) in enumerate(zip(sampled_classes_list, questions_list)):
            # 检查当前sample的数据是否有效
            if sampled_classes is None or questions is None:
                # 如果当前sample无效，使用原始conversation
                start_idx = offset_list[batch_idx] if batch_idx < len(offset_list) else 0
                end_idx = offset_list[batch_idx + 1] if batch_idx + 1 < len(offset_list) else len(conversation_list)
                for conv_idx in range(start_idx, end_idx):
                    if conv_idx < len(conversation_list):
                        part_conversation_list.append(conversation_list[conv_idx])
                        whole_conversation_list.append(conversation_list[conv_idx])
                continue
                
            # 遍历每个question及其对应的classes
            for q_idx, (question, classes_per_question) in enumerate(zip(questions, sampled_classes)):
                # 检查classes_per_question是否有效
                if classes_per_question is None or len(classes_per_question) == 0:
                    # 如果无效，使用原始conversation
                    conv_idx = offset_list[batch_idx] + q_idx if batch_idx < len(offset_list) else q_idx
                    if conv_idx < len(conversation_list):
                        part_conversation_list.append(conversation_list[conv_idx])
                        whole_conversation_list.append(conversation_list[conv_idx])
                    continue
                
                # 为每个class提取part和whole描述
                part_classes = []
                whole_classes = []
                original_classes = classes_per_question
                
                for sentence in classes_per_question:
                    sent_p, sent_w = "", ""
                    # print("sent_p",sent_p)
                    # print("sent_w",sent_w)
                    part_classes.append(sent_p)
                    whole_classes.append(sent_w)
                
                # 重新构建target和question（参考ReferSegDataset的逻辑）
                # Part conversation - 构建part_target用于问题和回答
                part_target = ''
                part_seg = []
                for i, text in enumerate(part_classes):
                    if i == len(part_classes) - 1:
                        part_seg.append('[SEG]')
                        part_target = part_target + (' and ' + text) if i != 0 else part_target + text
                    elif i == 0:
                        part_target += text
                        part_seg.append('[SEG]')
                        continue
                    else:
                        part_seg.append('[SEG]')
                        part_target += (', ' + text)

                if len(part_seg) > 1:
                    part1 = ', '.join(part_seg[:-1])
                    part2 = ' and ' + part_seg[-1]
                    part_seg_str = part1 + part2
                else:
                    part_seg_str = part_seg[0]

                # Whole conversation - 构建whole_target用于问题和回答
                whole_target = ''
                whole_seg = []
                for i, text in enumerate(whole_classes):
                    if i == len(whole_classes) - 1:
                        whole_seg.append('[SEG]')
                        whole_target = whole_target + (' and ' + text) if i != 0 else whole_target + text
                    elif i == 0:
                        whole_target += text
                        whole_seg.append('[SEG]')
                        continue
                    else:
                        whole_seg.append('[SEG]')
                        whole_target += (', ' + text)

                if len(whole_seg) > 1:
                    part1 = ', '.join(whole_seg[:-1])
                    part2 = ' and ' + whole_seg[-1]
                    whole_seg_str = part1 + part2
                else:
                    whole_seg_str = whole_seg[0]

                # 替换问题中的class_name
                # 找到原始问题中的referring expressions并替换
                original_target = ''
                for i, text in enumerate(original_classes):
                    if i == len(original_classes) - 1:
                        original_target = original_target + (' and ' + text) if i != 0 else original_target + text
                    elif i == 0:
                        original_target += text
                        continue
                    else:
                        original_target += (', ' + text)
                
                # 替换问题中的描述
                part_question = question.replace(original_target.lower(), part_target.lower())
                whole_question = question.replace(original_target.lower(), whole_target.lower())
                
                # 如果直接替换失败，尝试逐个替换
                if part_question == question:
                    for orig, part in zip(original_classes, part_classes):
                        part_question = part_question.replace(orig.lower(), part.lower())
                
                if whole_question == question:
                    for orig, whole in zip(original_classes, whole_classes):
                        whole_question = whole_question.replace(orig.lower(), whole.lower())

                # 生成答案（使用类似ReferSegDataset的模板）
                # Part answer - 强调专注于部分
                if len(part_classes) == 1:
                    part_answer = f"Sure, I can help you segment the specific part: {part_target.lower()}. [SEG]"
                else:
                    part_answer = f"I'll focus on segmenting the specific parts: {part_target.lower()}. {part_seg_str}"
                
                # Whole answer - 强调完整对象
                if len(whole_classes) == 1:
                    whole_answer = f"Sure, I can help you segment the complete object: {whole_target.lower()}. [SEG]"
                else:
                    whole_answer = f"I'll segment the complete objects: {whole_target.lower()}. {whole_seg_str}"

                # 构建conversation（参考ReferSegDataset的格式）
                conv = conversation_lib.default_conversation.copy()
                
                # Part conversation - 使用修改后的part_question
                conv.messages = []
                conv.append_message(conv.roles[0], part_question)
                conv.append_message(conv.roles[1], part_answer)
                part_conversation_list.append(conv.get_prompt())
                
                # Whole conversation - 使用修改后的whole_question
                conv.messages = []
                conv.append_message(conv.roles[0], whole_question)
                conv.append_message(conv.roles[1], whole_answer)
                whole_conversation_list.append(conv.get_prompt())
        
        return part_conversation_list, whole_conversation_list

    # 生成part和whole conversations
    part_conversation_list, whole_conversation_list = create_part_whole_conversations(
        sampled_classes_list, questions_list
    )
    # 处理image tokens
    if use_mm_start_end:
        # 处理原始conversations
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

    # 生成input_ids和attention_masks
    # 原始conversations
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

    # 处理targets (labels)
    conv = conversation_lib.conv_templates['chatml'].copy() if conv_type == "chatml" else conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    part_targets = part_input_ids.clone()
    whole_targets = whole_input_ids.clone()

    if conv_type == "llava_v1" or "chatml":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "

    # 处理原始targets
    for conversation, target in zip(conversation_list, targets):
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

    return {
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
            
            for sent in image_info['sentences']:
                sampled_sents.append(sent['sent'].strip().lower())
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
        )

