import argparse
import os
import shutil
import sys
import time
from functools import partial
from PIL import Image
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA_with_FGCLIP import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset
from utils.dataset_val_gen_data import ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

import matplotlib
import matplotlib.cm as cm
import pdb
import os
import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version" 
    ) # 模型权重
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="/data_16T/tc/huliwen/LISA-main/openaiclip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="RefCOCOm|val", type=str)
    parser.add_argument("--val_dataset", default="RefCOCOm|val", type=str)
    parser.add_argument("--dataset_dir", default="/data_16T/tc/huliwen/MMR-main/dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/data_16T/tc/huliwen/LISA-main/vision_pretrained/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--num_classes_per_question", default=3, type=int)
    parser.add_argument("--use_expand_question_list", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--val_json_name",default="testA_part_only.json")
    parser.add_argument("--fg_clip_mode", default="ipc", choices=["local", "ipc"], type=str)
    parser.add_argument("--fg_clip_root", default="/data_16T/tc/huliwen/FG-CLIP/fg_clip2", type=str)
    parser.add_argument("--fg_clip_python", default="", type=str)
    parser.add_argument("--fg_clip_worker_script", default="", type=str)
    parser.add_argument("--fg_clip_host", default="127.0.0.1", type=str)
    parser.add_argument("--fg_clip_port", default=29610, type=int)
    parser.add_argument("--fg_clip_authkey", default="m2sa_fgclip", type=str)
    parser.add_argument("--fg_clip_timeout", default=600.0, type=float)
    parser.add_argument("--fg_clip_worker_device", default="auto", type=str)
    parser.add_argument("--fg_clip_no_autostart", action="store_true", default=False)
    parser.add_argument("--similarity_backend", default="fg_clip", choices=["fg_clip", "dinov3"], type=str)
    parser.add_argument("--dinov3_root", default="/data_16T/tc/huliwen/dinov3_plus", type=str)
    parser.add_argument("--dinov3_long_side", default=756, type=int)
    parser.add_argument("--dinov3_ref_topk", default=1, type=int)
    parser.add_argument("--refine_method", default="legacy", choices=["legacy", "collab_single", "collab_iterative"], type=str)
    parser.add_argument("--refine_delta", default=1e-2, type=float)
    parser.add_argument("--refine_tmax", default=5, type=int)
    parser.add_argument("--dino_affinity_power", default=1.0, type=float)
    parser.add_argument("--refine_residual_weight", default=1.0, type=float)
    parser.add_argument("--whole_guidance_mode", default="contrast_part", choices=["soft_gate", "contrast_part"], type=str)
    parser.add_argument("--whole_overlap_thresh", default=0.8, type=float)
    parser.add_argument("--whole_value_high_thresh", default=0.97, type=float)
    parser.add_argument("--whole_value_low_thresh", default=0.03, type=float)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    fg_clip_worker_script = args.fg_clip_worker_script or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fg_clip_worker.py",
    )
    if args.fg_clip_worker_device == "auto":
        fg_clip_worker_device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        fg_clip_worker_device = args.fg_clip_worker_device

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "fg_clip_mode": args.fg_clip_mode,
        "fg_clip_root": args.fg_clip_root,
        "fg_clip_python": args.fg_clip_python or None,
        "fg_clip_worker_script": fg_clip_worker_script,
        "fg_clip_host": args.fg_clip_host,
        "fg_clip_port": args.fg_clip_port + int(args.local_rank),
        "fg_clip_authkey": args.fg_clip_authkey,
        "fg_clip_timeout": args.fg_clip_timeout,
        "fg_clip_start_worker": not args.fg_clip_no_autostart,
        "fg_clip_worker_device": fg_clip_worker_device,
        "similarity_backend": args.similarity_backend,
        "dinov3_root": args.dinov3_root,
        "dinov3_long_side": args.dinov3_long_side,
        "dinov3_ref_topk": args.dinov3_ref_topk,
        "refine_method": args.refine_method,
        "refine_delta": args.refine_delta,
        "refine_tmax": args.refine_tmax,
        "dino_affinity_power": args.dino_affinity_power,
        "refine_residual_weight": args.refine_residual_weight,
        "whole_guidance_mode": args.whole_guidance_mode,
        "whole_overlap_thresh": args.whole_overlap_thresh,
        "whole_value_high_thresh": args.whole_value_high_thresh,
        "whole_value_low_thresh": args.whole_value_low_thresh,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        # exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
            args.val_json_name
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if args.no_eval == False:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter

def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    intersection_meter_whole = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_whole = AverageMeter("Union", ":6.3f", Summary.SUM)
    intersection_meter_yes = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_yes = AverageMeter("Union", ":6.3f", Summary.SUM)
    intersection_meter_no = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter_no = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_whole_meter = AverageMeter("gIoU_whole", ":6.3f", Summary.SUM)
    acc_iou_fused_meter = AverageMeter("gIoU_whole", ":6.3f", Summary.SUM)
    acc_iou_nocc_meter = AverageMeter("gIoU_whole", ":6.3f", Summary.SUM)
    test_iou_meter = AverageMeter("ciou_whole", ":6.3f", Summary.SUM)
    model_engine.eval()
    sample = 0

    # 创建保存掩码和masked image的目录
    mask_save_dir_better = os.path.join('./results', f'better_414')
    mask_save_dir_worse = os.path.join('./results', f'worse_valp_0_1_close_1115')
    mask_save_dir_allthan20 = os.path.join('./results', f'better_valp_0_1_concat_ablation_1115')
    # masked_image_save_dir = os.path.join('./results', f'masked_images919')
    os.makedirs(mask_save_dir_better, exist_ok=True)
    os.makedirs(mask_save_dir_worse, exist_ok=True)
    os.makedirs(mask_save_dir_allthan20, exist_ok=True)
    # os.makedirs(masked_image_save_dir, exist_ok=True)

    # 统计掩码选择情况
    mask_selection_stats = {
        'pred_mask_used': 0,
        'whole_mask_used': 0,
        'total_samples': 0
    }

    ours_better, ori_better, equal = 0, 0, 0
    better_list = []
    worse_list = []

    batch_idx = 0
    import torch
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks_origin = output_dict["pred_masks_origin"]
        corrected_masks_logits = output_dict["corrected_masks"]

        part_pred_masks = output_dict["part_pred_masks"]
        fused_masks = output_dict["fused_masks"]
        whole_pred_masks = output_dict["whole_pred_masks"]
        ans_list = output_dict["ans_list"]
        image_paths = output_dict["image_paths"]
        texts_list = output_dict["text_list"]
        if args.val_dataset == "RefCOCOm|val":
            sent_ids_list = output_dict["sent_ids"]

        masks_list = output_dict["gt_masks"][0].int()
        output_list_origin = (pred_masks_origin[0] > 0).int()
        output_list_whole = (whole_pred_masks[0] > 0).int()
        output_list_part = (part_pred_masks[0] > 0).int()
        output_list_fused_masks = (fused_masks[0] > 0).int()
        corrected_masks = (corrected_masks_logits[0] > 0.0).int()
        
        # corrected_thresholds = corrected_scores.flatten(1).mean(dim=1).view(-1, 1, 1)
        # corrected_thresholds = torch.maximum(
        #     corrected_thresholds,
        #     torch.full_like(corrected_thresholds, 0.5),
        # )
        # corrected_masks = (corrected_scores > corrected_thresholds).int()

        scoremap_save_dir = os.path.join('./results', 'scoremap_viridis+mask')
        os.makedirs(scoremap_save_dir, exist_ok=True)
        for b_idx, (part_mask_batch, whole_mask_batch, baseline_batch, corrected_batch) in enumerate(
            zip(part_pred_masks, whole_pred_masks, pred_masks_origin, corrected_masks_logits)
        ):
            img_origin = input_dict["image_origins"][b_idx]
            path_name = str(image_paths[b_idx]) if b_idx < len(image_paths) else f"batch{batch_idx}"

            # gt_masks[b_idx]: [N, H, W]，texts_list 对应每个 mask 的 query
            for m_idx, single_logit in enumerate(part_mask_batch):
                gt_m = masks_list[m_idx] if m_idx < masks_list.shape[0] else None
                text_m = texts_list[b_idx * part_mask_batch.shape[0] + m_idx] \
                    if texts_list else ""
                baseline_logit = baseline_batch[m_idx] if m_idx < baseline_batch.shape[0] else None
                whole_logits = whole_mask_batch[0] if whole_mask_batch.dim() == 2 else \
                    whole_mask_batch[min(m_idx, whole_mask_batch.shape[0] - 1)]
                corrected_logit = corrected_batch[m_idx] if m_idx < corrected_batch.shape[0] else None

                if gt_m is None or baseline_logit is None or corrected_logit is None:
                    continue

                # save_path = os.path.join(
                #     scoremap_save_dir,
                #     f"{path_name}_part_m{m_idx}_scoremap.png"
                # )
                # visualize_part_scoremap_viridis(
                #     baseline_score_map=baseline_logit,
                #     whole_score_map=whole_logits,
                #     part_score_map=single_logit,
                #     corrected_score_map=corrected_logit,
                #     baseline_mask=output_list_origin[m_idx] if m_idx < output_list_origin.shape[0] else None,
                #     whole_mask=output_list_whole[m_idx] if m_idx < output_list_whole.shape[0] else None,
                #     part_mask=output_list_part[m_idx] if m_idx < output_list_part.shape[0] else None,
                #     corrected_mask=corrected_masks[m_idx] if m_idx < corrected_masks.shape[0] else None,
                #     image_origin=img_origin,
                #     gt_mask=gt_m,
                #     text=text_m,
                #     save_path=save_path,
                #     alpha=0.55,
                # )
        # assert len(output_list_part_sam) == 1
        assert len(whole_pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        intersection_whole, union_whole, acc_iou_whole = 0.0, 0.0, 0.0
        intersection_yes, union_yes, acc_iou_yes = 0.0, 0.0, 0.0
        intersection_no, union_no, acc_iou_no = 0.0, 0.0, 0.0
        intersection_fuse, union_fuse, acc_iou_fuse = 0.0, 0.0, 0.0

        sample = sample + masks_list.shape[0]

        # output_list_whole = torch.stack(output_list_whole)
        if len(corrected_masks) == 0:
            batch_idx += 1
            continue
        # 比较pred_masks_origin和pred_masks的IoU
        for idx, (mask_i, output_i, output_i_origin, output_i_whole, output_i_part, output_i_fused) in enumerate(zip(
                masks_list, corrected_masks, output_list_origin, output_list_whole, output_list_part,
                output_list_fused_masks)):
            # 计算ours (output_list_new)的IoU
            intersection_ours, union_ours, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            iou_ours = (intersection_ours[1] / (union_ours[1] + 1e-5)).cpu().item()

            # 计算origin的IoU
            intersection_ori, union_ori, _ = intersectionAndUnionGPU(
                output_i_origin.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )

            intersection_fused, union_fused, _ = intersectionAndUnionGPU(
                output_i_fused.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )

            iou_ori = (intersection_ori[1] / (union_ori[1] + 1e-5)).cpu().item()

            if args.val_dataset == 'RefCOCOm|val':
                image_path = str(image_paths[0]) + "_" + "--" + ans_list[idx] + "--" + str(sent_ids_list[idx]) + "--" + \
                             texts_list[idx]
            else:
                image_path = str(image_paths[0]) + "_" + "--" + ans_list[idx] + "--" + texts_list[idx]

            if iou_ours > iou_ori + 1e-5:  # ours更好
                ours_better += 1
                better_list.append({
                    "image_path": image_path,
                    "iou_ours": float(iou_ours),
                    "iou_origin": float(iou_ori),
                    "diff": float(iou_ours - iou_ori)
                })
                # if iou_sam20 + 1e-5 < iou_ours:
                #     save_pred_masks_as_png([mask_i,output_i_origin,output_i,output_i_whole,output_i_part,output_i_fused,output_i_sam20], image_path, mask_save_dir_allthan20)

                # save_pred_masks_as_png(
                #     [mask_i, output_i_origin, output_i, output_i_whole, output_i_part, output_i_fused], image_path,
                #     mask_save_dir_better)
            elif iou_ori > iou_ours + 1e-5:  # origin更好
                ori_better += 1
                worse_list.append({
                    "image_path": image_path,
                    "iou_ours": float(iou_ours),
                    "iou_origin": float(iou_ori),
                    "diff": float(iou_ori - iou_ours)
                })

                # save_pred_masks_as_png([mask_i,output_i_origin,output_i,output_i_whole,output_i_part,output_i_fused,output_i_sam20], image_path, mask_save_dir_worse)
            else:  # 相等
                equal += 1

            # 原有的累加逻辑
            intersection += intersection_ours
            union += union_ours
            acc_iou += intersection_ours / (union_ours + 1e-5)
            acc_iou[union_ours == 0] += 1.0

            intersection_fuse += intersection_fused
            union_fuse += union_fused
            acc_iou_fuse += intersection_fused / (union_fused + 1e-5)
            acc_iou_fuse[union_fused == 0] += 1.0

            if ans_list[idx] == "no":
                intersection_no += intersection_ours
                union_no += union_ours
                intersection_yes += intersection_ours
                union_yes += union_ours
                intersection_yes -= intersection_ours
                union_yes -= union_ours
            else:
                intersection_yes += intersection_ours
                union_yes += union_ours
                intersection_no += intersection_ours
                union_no += union_ours
                intersection_no -= intersection_ours
                union_no -= union_ours

            test_iou = intersection_ours / (union_ours + 1e-5)
            test_iou[union_ours == 0] += 1.0
            test_iou = test_iou.cpu().numpy()
            test_iou_meter.update(test_iou)

        for mask_i, output_i in zip(masks_list, output_list_whole):
            test_iou = 0.0
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )

            intersection_whole += intersection_i
            union_whole += union_i
            acc_iou_whole += intersection_i / (union_i + 1e-5)
            acc_iou_whole[union_i == 0] += 1.0  # no-object target

        # shape [2, 1]
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()

        intersection_whole, union_whole = intersection_whole.cpu().numpy(), union_whole.cpu().numpy()

        intersection_yes, union_yes = intersection_yes.cpu().numpy(), union_yes.cpu().numpy()

        intersection_no, union_no = intersection_no.cpu().numpy(), union_no.cpu().numpy()

        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        acc_iou_whole = acc_iou_whole.cpu().numpy() / masks_list.shape[0]
        acc_iou_fuse = acc_iou_fuse.cpu().numpy() / masks_list.shape[0]

        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0]),
        acc_iou_whole_meter.update(acc_iou_whole, n=masks_list.shape[0]),
        acc_iou_fused_meter.update(acc_iou_fuse, n=masks_list.shape[0]),
        intersection_meter_whole.update(intersection_whole), union_meter_whole.update(
            union_whole
        ), intersection_meter_yes.update(intersection_yes), union_meter_yes.update(
            union_yes
        ), intersection_meter_no.update(intersection_no), union_meter_no.update(
            union_no
        )

        batch_idx += 1
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    intersection_meter_whole.all_reduce()
    union_meter_whole.all_reduce()
    intersection_meter_yes.all_reduce()
    union_meter_yes.all_reduce()
    intersection_meter_no.all_reduce()
    union_meter_no.all_reduce()
    acc_iou_meter.all_reduce()
    test_iou_meter.all_reduce()
    acc_iou_whole_meter.all_reduce()
    acc_iou_fused_meter.all_reduce()
    acc_iou_nocc_meter.all_reduce()

    # 保存比较结果到JSON文件
    # results_dir = './duibi'
    # os.makedirs(results_dir, exist_ok=True)

    # with open(os.path.join(results_dir, 'ours_better_valp_0_1_fix.json'), 'w') as f:
    #     json.dump(better_list, f, indent=2)

    # with open(os.path.join(results_dir, 'ori_better_valp_0_1_fix.json'), 'w') as f:
    #     json.dump(worse_list, f, indent=2)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    iou_class_whole = intersection_meter_whole.sum / (union_meter_whole.sum + 1e-10)
    iou_class_yes = intersection_meter_yes.sum / (union_meter_yes.sum + 1e-10)
    iou_class_no = intersection_meter_no.sum / (union_meter_no.sum + 1e-10)
    miou = test_iou_meter.avg[1]
    # print(miou)
    ciou = iou_class[1]
    ciou_whole = iou_class_whole[1]
    ciou_yes = iou_class_yes[1]
    ciou_no = iou_class_no[1]
    giou = acc_iou_meter.avg[1]
    giou_whole = acc_iou_whole_meter.avg[1]
    giou_fused = acc_iou_fused_meter.avg[1]

    # miou = np.mean(iou_class)
    global_pred_mask_used = mask_selection_stats['pred_mask_used']
    global_whole_mask_used = mask_selection_stats['whole_mask_used']
    print(f"使用pred_mask: {global_pred_mask_used}")
    print(f"使用whole_mask: {global_whole_mask_used}")
    if args.local_rank == 0:
        print("比较结果统计:")
        print(f"Ours更好的样本数: {ours_better}")
        print(f"Origin更好的样本数: {ori_better}")
        print(f"相等的样本数: {equal}")
        print(f"总样本数: {ours_better + ori_better + equal}")
        print("=" * 50)
        print("miou: {:4f}".format(miou))
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        print("giou of whole:{:.4f}".format(giou_whole))
        print("giou of fused:{:.4f}".format(giou_fused))

        # print("ciou of whole:{:.4f}".format(ciou_whole))
        # print("ciou of yes:{:.4f}".format(ciou_yes))
        # print("ciou of no:{:.4f}".format(ciou_no))
        # print(f"Better samples saved to: {os.path.join(results_dir, 'ours_better_samples.json')}")
        # print(f"Worse samples saved to: {os.path.join(results_dir, 'ori_better_samples.json')}")
    # print(f"Masked images saved to: {masked_image_save_dir}")

    return giou, ciou


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def calculate_mask_iou(mask1, mask2):
    """
    计算两个掩码之间的IoU
    Args:
        mask1, mask2: shape [H, W] 的二值掩码
    Returns:
        iou: float值
    """
    # 确保是二值掩码
    mask1_bin = (mask1 > 0).float()
    mask2_bin = (mask2 > 0).float()
    
    # 计算交集和并集
    intersection = torch.sum(mask1_bin * mask2_bin)
    union = torch.sum(mask1_bin) + torch.sum(mask2_bin) - intersection
    
    # 避免除零
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def save_pred_masks_as_png(pred_masks, image_numbers,sent_ids, save_dir):
    """
    将预测的掩码保存为PNG图片，文件名使用给定的数字编号

    Args:
        pred_masks: list of torch.Tensor，每个tensor形状为 [num_masks, H, W]
        image_numbers: list of int，对应每组mask的图像编号
        save_dir: str，保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for group_idx, (mask_group, num) in enumerate(zip(pred_masks, image_numbers)):
        mask_group_np = mask_group.cpu().numpy()  # [num_masks, H, W]

        for mask_idx in range(mask_group_np.shape[0]):
            single_mask = mask_group_np[mask_idx]
            binary_mask = (single_mask > 0).astype(np.uint8) * 255  # 二值化
            pil_image = Image.fromarray(binary_mask, mode='L')

            # ✅ 用数字编号命名，如 25_mask_00.png
            filename = f"{num}_mask_{sent_ids[mask_idx]}.png"
            filepath = os.path.join(save_dir, filename)
            pil_image.save(filepath)

def save_masked_images(images, pred_masks, batch_idx, save_dir):
    """
    保存masked images，即保留掩码区域内的图像，其余区域填0
    
    Args:
        images: torch.Tensor, 形状为 [batch_size, C, H, W]
        pred_masks: list of torch.Tensor, 每个tensor形状为 [num_masks, H, W]
        batch_idx: int, 当前batch的索引
        save_dir: str, 保存目录
    """
    images = images.to(torch.float)
    # 将images转换为numpy并调整维度顺序 [batch_size, H, W, C]
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    
    # 如果图像是归一化的 (0-1范围)，转换为0-255范围
    if images_np.max() <= 1.0:
        images_np = (images_np * 255).astype(np.uint8)
    else:
        images_np = images_np.astype(np.uint8)
    
    for group_idx, mask_group in enumerate(pred_masks):
        # mask_group shape: [num_masks, H, W]
        mask_group_np = mask_group.cpu().numpy()
        
        for mask_idx in range(mask_group_np.shape[0]):
            # 获取单个掩码 [H, W]
            single_mask = mask_group_np[mask_idx]
            
            # 创建二值掩码
            binary_mask = (single_mask > 0)
            
            # 假设batch中第一张图像对应当前掩码
            # 如果有多张图像，可能需要根据实际情况调整索引
            if group_idx < len(images_np):
                image = images_np[group_idx]  # [H, W, C]
                
                # 应用掩码：保留掩码区域，其余填0
                masked_image = image.copy()
                masked_image[~binary_mask] = 0  # 掩码外区域填0
                
                # 转换为PIL图像
                if masked_image.shape[2] == 3:  # RGB图像
                    pil_image = Image.fromarray(masked_image, mode='RGB')
                elif masked_image.shape[2] == 1:  # 灰度图像
                    pil_image = Image.fromarray(masked_image.squeeze(2), mode='L')
                else:
                    # 如果通道数不是1或3，转换为RGB
                    if masked_image.shape[2] > 3:
                        masked_image = masked_image[:, :, :3]
                    pil_image = Image.fromarray(masked_image, mode='RGB')
                
                # 生成文件名
                filename = f"batch_{batch_idx:04d}_group_{group_idx}_masked_{mask_idx:02d}.png"
                filepath = os.path.join(save_dir, filename)
                
                # 保存图像
                pil_image.save(filepath)

def visualize_part_scoremap_viridis(
    baseline_score_map: torch.Tensor,
    whole_score_map: torch.Tensor,
    part_score_map: torch.Tensor,
    corrected_score_map: torch.Tensor,
    baseline_mask: torch.Tensor,
    whole_mask: torch.Tensor,
    part_mask: torch.Tensor,
    corrected_mask: torch.Tensor,
    image_origin,
    gt_mask: torch.Tensor,
    text: str,
    save_path: str,
    alpha: float = 0.55,
):
    def _to_score_np(score_map):
        if isinstance(score_map, torch.Tensor):
            score_np = torch.sigmoid(score_map).float().cpu().numpy()
        else:
            score_np = score_map.astype(np.float32)
        if score_np.ndim != 2:
            score_np = score_np.squeeze()
        if score_np.ndim != 2:
            score_np = score_np[0]
        return score_np.astype(np.float32)

    def _to_mask_np(mask, target_hw):
        if mask is None:
            return np.zeros(target_hw, dtype=np.uint8)
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        if mask_np.ndim != 2:
            mask_np = np.squeeze(mask_np)
        if mask_np.ndim != 2:
            mask_np = mask_np[0]
        mask_np = (mask_np > 0).astype(np.uint8)
        if mask_np.shape != target_hw:
            mask_np = cv2.resize(mask_np, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        return mask_np

    baseline_np = _to_score_np(baseline_score_map)
    H, W = baseline_np.shape
    whole_np = _to_score_np(whole_score_map)
    part_np = _to_score_np(part_score_map)
    corrected_np = _to_score_np(corrected_score_map)

    if whole_np.shape != (H, W):
        whole_np = cv2.resize(whole_np, (W, H), interpolation=cv2.INTER_LINEAR)
    if part_np.shape != (H, W):
        part_np = cv2.resize(part_np, (W, H), interpolation=cv2.INTER_LINEAR)
    if corrected_np.shape != (H, W):
        corrected_np = cv2.resize(corrected_np, (W, H), interpolation=cv2.INTER_LINEAR)

    if isinstance(image_origin, Image.Image):
        img_np = np.array(image_origin.convert("RGB"))
    else:
        img_np = np.array(image_origin)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

    if img_np.shape[:2] != (H, W):
        img_np = np.array(Image.fromarray(img_np).resize((W, H), Image.BILINEAR))

    img_f = img_np.astype(np.float32) / 255.0

    if isinstance(gt_mask, torch.Tensor):
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
    else:
        gt_np = gt_mask.astype(np.uint8)
    if gt_np.shape != (H, W):
        gt_np = cv2.resize(gt_np, (W, H), interpolation=cv2.INTER_NEAREST)
    gt_vis = (gt_np * 255).astype(np.uint8)
    gt_rgb = np.stack([gt_vis] * 3, axis=-1)

    baseline_mask_np = _to_mask_np(baseline_mask, (H, W))
    whole_mask_np = _to_mask_np(whole_mask, (H, W))
    part_mask_np = _to_mask_np(part_mask, (H, W))
    corrected_mask_np = _to_mask_np(corrected_mask, (H, W))

    def _make_mask_panel(mask_np):
        mask_vis = (mask_np * 255).astype(np.uint8)
        return np.stack([mask_vis] * 3, axis=-1)

    cmap = matplotlib.colormaps["viridis"]

    def _make_score_panel(score_np):
        score_rgb = (cmap(score_np)[:, :, :3] * 255).astype(np.uint8)
        overlay_f = cmap(score_np)[:, :, :3]
        overlay = np.clip((1 - alpha) * img_f + alpha * overlay_f, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        return score_rgb, overlay

    baseline_rgb, _ = _make_score_panel(baseline_np)
    whole_rgb, _ = _make_score_panel(whole_np)
    part_rgb, _ = _make_score_panel(part_np)
    corrected_rgb, _ = _make_score_panel(corrected_np)

    baseline_mask_rgb = _make_mask_panel(baseline_mask_np)
    whole_mask_rgb = _make_mask_panel(whole_mask_np)
    part_mask_rgb = _make_mask_panel(part_mask_np)
    corrected_mask_rgb = _make_mask_panel(corrected_mask_np)

    panels = [
        ("original", img_np, img_np, None, None),
        ("gt", gt_rgb, gt_rgb, None, gt_np.astype(np.float32)),
        ("baseline", baseline_rgb, baseline_mask_rgb, baseline_np, baseline_mask_np.astype(np.float32)),
        ("whole", whole_rgb, whole_mask_rgb, whole_np, whole_mask_np.astype(np.float32)),
        ("part", part_rgb, part_mask_rgb, part_np, part_mask_np.astype(np.float32)),
        ("correct", corrected_rgb, corrected_mask_rgb, corrected_np, corrected_mask_np.astype(np.float32)),
    ]

    score_row = np.concatenate([score_panel for _, score_panel, _, _, _ in panels], axis=1)
    mask_row = np.concatenate([mask_panel for _, _, mask_panel, _, _ in panels], axis=1)
    total_w = score_row.shape[1]
    bar_h = 80
    text_bar = np.full((bar_h, total_w, 3), 30, dtype=np.uint8)
    canvas = np.concatenate([text_bar, score_row, mask_row], axis=0)

    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (label, _, _, score_np, mask_np) in enumerate(panels):
        cx = idx * W
        draw.text((cx + 6, 4), label, fill=(200, 200, 200), font=font)
        if score_np is not None:
            stats = f"min {score_np.min():.3f} max {score_np.max():.3f} mean {score_np.mean():.3f}"
            draw.text((cx + 6, 24), stats, fill=(160, 220, 255), font=font)
        if mask_np is not None:
            fg_ratio = float(mask_np.mean())
            draw.text((cx + 6, 44), f"mask fg {fg_ratio:.3f}", fill=(180, 255, 180), font=font)

    max_chars = total_w // 9
    display_text = text if len(text) <= max_chars else text[:max_chars - 3] + "..."
    draw.text((6, 62), f"text: {display_text}", fill=(255, 220, 100), font=font)
    draw.text((6, bar_h + 6), "score map", fill=(255, 255, 255), font=font)
    draw.text((6, bar_h + H + 6), "mask", fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    canvas_pil.save(save_path)



if __name__ == "__main__":
    main(sys.argv[1:])
