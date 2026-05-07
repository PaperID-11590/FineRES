from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from pathlib import Path
import re
from .sam_mask_loader import SAMMaskLoader

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .segment_anything2 import sam_model_registry, SamAutomaticMaskGenerator
import transformers

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from PIL import Image

try:
    from .fg_clip_ipc import FGClipIPCClient
except ImportError:
    from fg_clip_ipc import FGClipIPCClient


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class M2SAMetaModel:
    def __init__(
        self,
        config,
        **kwargs
    ):
        super(M2SAMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_M2SA_modules(self.config)

    def initialize_M2SA_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)

        for param in self.visual_model.parameters():
            param.requires_grad = False

        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class M2SAModel(M2SAMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(M2SAModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class M2SAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openaiclip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.local_rank = kwargs.pop("local_rank", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.clip_image_processor = CLIPImageProcessor.from_pretrained("openaiclip-vit-large-patch14")
        self.fg_clip_mode = kwargs.pop("fg_clip_mode", "ipc")
        self.fg_clip_python = kwargs.pop("fg_clip_python", None)
        self.fg_clip_worker_script = kwargs.pop("fg_clip_worker_script", None)
        self.fg_clip_host = kwargs.pop("fg_clip_host", "127.0.0.1")
        self.fg_clip_port = int(kwargs.pop("fg_clip_port", 29610))
        self.fg_clip_authkey = kwargs.pop("fg_clip_authkey", "m2sa_fgclip")
        self.fg_clip_timeout = float(kwargs.pop("fg_clip_timeout", 600.0))
        self.fg_clip_start_worker = bool(kwargs.pop("fg_clip_start_worker", True))
        self.fg_clip_worker_device = kwargs.pop("fg_clip_worker_device", None)
        self.similarity_backend = kwargs.pop("similarity_backend", "fg_clip")

        # ── FG-CLIP 配置 ──────────────────────────────────────────────────────
        # fg_clip_root 可通过 kwargs 传入，否则使用默认路径
        self.fg_clip_root = kwargs.pop(
            "fg_clip_root",
            "/data_16T/tc/huliwen/FG-CLIP/fg_clip2"
        )
        self.dinov3_root = kwargs.pop(
            "dinov3_root",
            "/data_16T/tc/huliwen/dinov3_plus"
        )
        self.dinov3_long_side = int(kwargs.pop("dinov3_long_side", 756))
        self.dinov3_ref_topk = int(kwargs.pop("dinov3_ref_topk", 1))
        self.refine_method = kwargs.pop("refine_method", "legacy")
        self.refine_delta = float(kwargs.pop("refine_delta", 1e-3))
        self.refine_tmax = int(kwargs.pop("refine_tmax", 5))
        self.dino_affinity_power = float(kwargs.pop("dino_affinity_power", 1.0))
        self.refine_residual_weight = float(kwargs.pop("refine_residual_weight", 1.0))
        self.whole_guidance_mode = kwargs.pop("whole_guidance_mode", "soft_gate")
        self.whole_contrast_weight = float(kwargs.pop("whole_contrast_weight", 0.5))
        self.whole_overlap_thresh = float(kwargs.pop("whole_overlap_thresh", 0.8))
        self.whole_value_high_thresh = float(kwargs.pop("whole_value_high_thresh", 0.97))
        self.whole_value_low_thresh = float(kwargs.pop("whole_value_low_thresh", 0.03))
        # Dual Correction 超参数（均可由 kwargs 覆盖）
        self.dc_tau_c        = kwargs.pop("dc_tau_c",        0.5)   # 高置信核阈值
        self.dc_kernel_size  = kwargs.pop("dc_kernel_size",  5)     # 局部传播卷积核大小（奇数）
        self.dc_eps          = kwargs.pop("dc_eps",          1e-6)  # 数值稳定量
        self.dc_fg_clip_max_patches = kwargs.pop("dc_fg_clip_max_patches", 4096)
        self.dc_fg_clip_resize      = kwargs.pop("dc_fg_clip_resize",      2048)
        # ─────────────────────────────────────────────────────────────────────

        super().__init__(config)
        self.model = M2SAModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self._fg_clip_client = None

        # FG-CLIP 延迟加载（首次调用时初始化）
        self._fg_clip_model            = None
        self._fg_clip_image_processor  = None
        self._fg_clip_tokenizer        = None
        self._dinov3_model             = None
        self._dinov3_image_processor   = None
        self._dinov3_patch_size        = 14

    # =========================================================================
    # FG-CLIP 延迟加载
    # =========================================================================
    def _ensure_fg_clip_loaded(self):
        """首次调用时加载 FG-CLIP，后续复用缓存。FG-CLIP 参数全程冻结。"""
        if self._fg_clip_model is not None:
            return
        device = next(self.parameters()).device
        print(f"[M2SA] Loading FG-CLIP from {self.fg_clip_root} ...")
        self._fg_clip_model = (
            AutoModelForCausalLM.from_pretrained(
                self.fg_clip_root, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        for p in self._fg_clip_model.parameters():
            p.requires_grad = False
        self._fg_clip_image_processor = AutoImageProcessor.from_pretrained(self.fg_clip_root)
        self._fg_clip_tokenizer       = AutoTokenizer.from_pretrained(self.fg_clip_root)
        print("[M2SA] FG-CLIP loaded.")

    # =========================================================================
    # FG-CLIP：计算单张图像 × 单条文本的像素级相似度图 At
    #   返回 shape [H_mask, W_mask] 的 float32 Tensor（与 mask 分辨率一致）
    # =========================================================================
    @torch.no_grad()
    def _compute_fg_clip_similarity_map(
        self,
        image_pil: Image.Image,
        text_query: str,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        利用 FG-CLIP 计算 text_query 与 image_pil 每个 patch 的余弦相似度，
        然后双线性插值到 (target_h, target_w)。

        Args:
            image_pil   : PIL.Image，原始图像
            text_query  : str，part 级别的文本描述
            target_h/w  : 目标分辨率（与 mask 一致）

        Returns:
            sim_map : Tensor [target_h, target_w]，值域 [-1, 1]
        """
        self._ensure_fg_clip_loaded()
        device = next(self._fg_clip_model.parameters()).device

        # ── 图像预处理（短边缩放 + patch tokenize）─────────────────────────
        img_w, img_h = image_pil.size
        short = min(img_w, img_h)
        if short < self.dc_fg_clip_resize:
            scale     = self.dc_fg_clip_resize / short
            image_pil = image_pil.resize(
                (int(img_w * scale), int(img_h * scale)), Image.BILINEAR
            )

        image_input = self._fg_clip_image_processor(
            images=image_pil,
            max_num_patches=self.dc_fg_clip_max_patches,
            return_tensors="pt",
        ).to(device)

        # ── 提取稠密图像特征 ───────────────────────────────────────────────
        dense_feat = self._fg_clip_model.get_image_dense_feature(**image_input)  # [1, N, D]

        spatial = image_input["spatial_shapes"][0]
        feat_h  = int(spatial[0].item())
        feat_w  = int(spatial[1].item())
        n_tok   = feat_h * feat_w

        dense_feat = dense_feat[0, :n_tok]                              # [H_f*W_f, D]
        dense_feat = dense_feat / (dense_feat.norm(p=2, dim=-1, keepdim=True) + self.dc_eps)

        # ── 提取文本特征 ───────────────────────────────────────────────────
        caption_input = self._fg_clip_tokenizer(
            [text_query.lower()],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_feat = self._fg_clip_model.get_text_features(**caption_input, walk_type="box")  # [1, D]
        text_feat = text_feat / (text_feat.norm(p=2, dim=-1, keepdim=True) + self.dc_eps)

        # ── 余弦相似度 → 空间图 ────────────────────────────────────────────
        sim = (dense_feat @ text_feat.T).squeeze(-1)                    # [H_f*W_f]
        sim_map = sim.reshape(1, 1, feat_h, feat_w)                     # [1,1,H_f,W_f]

        # ── 双线性上采样到 mask 分辨率 ────────────────────────────────────
        sim_map_up = F.interpolate(
            sim_map, size=(target_h, target_w), mode="bilinear", align_corners=False
        ).squeeze()                                                      # [target_h, target_w]

        return sim_map_up.float()

    # =========================================================================
    # Algorithm 1: Dual Correction for Part Score Map
    # =========================================================================
    def _ensure_fg_clip_loaded(self):
        if self.fg_clip_mode == "ipc":
            if self._fg_clip_client is not None:
                return
            worker_script = self.fg_clip_worker_script
            if worker_script is None:
                worker_script = str(Path(__file__).resolve().with_name("fg_clip_worker.py"))
            self._fg_clip_client = FGClipIPCClient(
                host=self.fg_clip_host,
                port=self.fg_clip_port,
                authkey=self.fg_clip_authkey,
                timeout=self.fg_clip_timeout,
            )
            self._fg_clip_client.connect_or_start(
                model_root=self.fg_clip_root,
                python_executable=self.fg_clip_python,
                worker_script=worker_script,
                device=self.fg_clip_worker_device,
                auto_start=self.fg_clip_start_worker,
                dinov3_root=self.dinov3_root,
            )
            return

        if self._fg_clip_model is not None:
            return

        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

        device = next(self.parameters()).device
        print(f"[M2SA] Loading FG-CLIP locally from {self.fg_clip_root} ...")
        self._fg_clip_model = (
            AutoModelForCausalLM.from_pretrained(
                self.fg_clip_root, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        for p in self._fg_clip_model.parameters():
            p.requires_grad = False
        self._fg_clip_image_processor = AutoImageProcessor.from_pretrained(self.fg_clip_root)
        self._fg_clip_tokenizer = AutoTokenizer.from_pretrained(self.fg_clip_root)
        print("[M2SA] FG-CLIP loaded.")

    @torch.no_grad()
    def _compute_fg_clip_similarity_maps(
        self,
        image_pil: Image.Image,
        text_queries: List[str],
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if len(text_queries) == 0:
            return torch.empty(0, target_h, target_w, dtype=torch.float32)

        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(np.asarray(image_pil).astype(np.uint8))
        image_pil = image_pil.convert("RGB")
        normalized_texts = [text or "" for text in text_queries]

        self._ensure_fg_clip_loaded()

        if self.fg_clip_mode == "ipc":
            image_array = np.ascontiguousarray(np.array(image_pil, dtype=np.uint8))
            sim_maps = self._fg_clip_client.compute_similarity_maps(
                image_array=image_array,
                text_queries=normalized_texts,
                target_h=target_h,
                target_w=target_w,
                resize_short=self.dc_fg_clip_resize,
                max_num_patches=self.dc_fg_clip_max_patches,
            )
            return torch.from_numpy(sim_maps)

        device = next(self._fg_clip_model.parameters()).device
        img_w, img_h = image_pil.size
        short = min(img_w, img_h)
        if short < self.dc_fg_clip_resize:
            scale = self.dc_fg_clip_resize / short
            image_pil = image_pil.resize(
                (int(img_w * scale), int(img_h * scale)), Image.BILINEAR
            )

        image_input = self._fg_clip_image_processor(
            images=image_pil,
            max_num_patches=self.dc_fg_clip_max_patches,
            return_tensors="pt",
        ).to(device)
        dense_feat = self._fg_clip_model.get_image_dense_feature(**image_input)

        spatial = image_input["spatial_shapes"][0]
        feat_h = int(spatial[0].item())
        feat_w = int(spatial[1].item())
        n_tok = feat_h * feat_w

        dense_feat = dense_feat[0, :n_tok]
        dense_feat = dense_feat / (dense_feat.norm(p=2, dim=-1, keepdim=True) + self.dc_eps)

        caption_input = self._fg_clip_tokenizer(
            [text.lower() for text in normalized_texts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_feat = self._fg_clip_model.get_text_features(**caption_input, walk_type="box")
        text_feat = text_feat / (text_feat.norm(p=2, dim=-1, keepdim=True) + self.dc_eps)

        sim = dense_feat @ text_feat.T
        sim_maps = sim.transpose(0, 1).reshape(len(normalized_texts), 1, feat_h, feat_w)
        sim_maps_up = F.interpolate(
            sim_maps,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return sim_maps_up.float()

    @torch.no_grad()
    def _compute_fg_clip_similarity_map(
        self,
        image_pil: Image.Image,
        text_query: str,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        return self._compute_fg_clip_similarity_maps(
            image_pil=image_pil,
            text_queries=[text_query],
            target_h=target_h,
            target_w=target_w,
        )[0]

    def _ensure_dinov3_loaded(self):
        if self._dinov3_model is not None:
            return

        from transformers import AutoImageProcessor, AutoModel

        self._dinov3_image_processor = AutoImageProcessor.from_pretrained(self.dinov3_root)
        self._dinov3_model = AutoModel.from_pretrained(self.dinov3_root).to(
            next(self.parameters()).device
        ).eval()
        for param in self._dinov3_model.parameters():
            param.requires_grad = False
        if hasattr(self._dinov3_model.config, "patch_size"):
            self._dinov3_patch_size = int(self._dinov3_model.config.patch_size)
        print(
            f"[M2SA] DINOv3 loaded from {self.dinov3_root} "
            f"(patch_size={self._dinov3_patch_size})"
        )

    def _resize_image_to_dino_grid(self, image_pil: Image.Image) -> Image.Image:
        image_pil = image_pil.convert("RGB")
        patch = self._dinov3_patch_size
        width, height = image_pil.size
        scale = float(self.dinov3_long_side) / max(height, width)
        new_h = max(patch, int(round(height * scale)))
        new_w = max(patch, int(round(width * scale)))
        new_h = ((new_h + patch - 1) // patch) * patch
        new_w = ((new_w + patch - 1) // patch) * patch
        return image_pil.resize((new_w, new_h), Image.LANCZOS)

    @torch.no_grad()
    def _compute_dinov3_similarity_maps(
        self,
        image_pil: Image.Image,
        part_score_maps: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if part_score_maps.numel() == 0:
            return torch.empty(0, target_h, target_w, dtype=torch.float32)

        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(np.asarray(image_pil).astype(np.uint8))
        image_pil = image_pil.convert("RGB")

        if self.fg_clip_mode == "ipc":
            self._ensure_fg_clip_loaded()
            image_array = np.ascontiguousarray(np.array(image_pil, dtype=np.uint8))
            score_maps = part_score_maps.detach().cpu().float().numpy()
            sim_maps = self._fg_clip_client.compute_similarity_maps(
                image_array=image_array,
                text_queries=None,
                target_h=target_h,
                target_w=target_w,
                resize_short=self.dc_fg_clip_resize,
                max_num_patches=self.dc_fg_clip_max_patches,
                backend="dinov3",
                score_maps=score_maps,
                dino_long_side=self.dinov3_long_side,
                dino_ref_topk=self.dinov3_ref_topk,
            )
            return torch.from_numpy(sim_maps)

        self._ensure_dinov3_loaded()
        device = next(self._dinov3_model.parameters()).device
        image_pil = self._resize_image_to_dino_grid(image_pil)

        inputs = self._dinov3_image_processor(
            images=image_pil,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(device)
        outputs = self._dinov3_model(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state

        img_w, img_h = image_pil.size
        hp = img_h // self._dinov3_patch_size
        wp = img_w // self._dinov3_patch_size
        expected = hp * wp
        special_tokens = hidden.shape[1] - expected
        if special_tokens < 0:
            raise ValueError(
                f"DINO token mismatch: total={hidden.shape[1]}, expected patches={expected}"
            )
        patch_tokens = hidden[:, special_tokens:, :]
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"DINO patch token mismatch: expected={expected}, actual={patch_tokens.shape[1]}"
            )

        patch_features = F.normalize(patch_tokens[0], dim=-1)
        score_tensor = F.interpolate(
            part_score_maps.unsqueeze(1).to(device=device, dtype=patch_features.dtype),
            size=(hp, wp),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        flat_scores = score_tensor.reshape(score_tensor.shape[0], -1)

        topk = max(1, min(int(self.dinov3_ref_topk), flat_scores.shape[1]))
        if topk == 1:
            anchor_indices = flat_scores.argmax(dim=1)
            ref_features = patch_features[anchor_indices]
        else:
            top_values, top_indices = torch.topk(flat_scores, k=topk, dim=1)
            top_features = patch_features[top_indices]
            weights = torch.softmax(top_values, dim=1).unsqueeze(-1)
            ref_features = (top_features * weights).sum(dim=1)
            ref_features = F.normalize(ref_features, dim=-1)

        sim = patch_features @ ref_features.T
        sim_maps = sim.transpose(0, 1).reshape(score_tensor.shape[0], 1, hp, wp)
        sim_maps_up = F.interpolate(
            sim_maps,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        return sim_maps_up.float()

    @torch.no_grad()
    def _compute_dino_patch_features(
        self,
        image_pil: Image.Image,
    ):
        if not isinstance(image_pil, Image.Image):
            image_pil = Image.fromarray(np.asarray(image_pil).astype(np.uint8))
        image_pil = image_pil.convert("RGB")

        if self.fg_clip_mode == "ipc":
            self._ensure_fg_clip_loaded()
            image_array = np.ascontiguousarray(np.array(image_pil, dtype=np.uint8))
            patch_features, hp, wp = self._fg_clip_client.compute_dino_patch_features(
                image_array=image_array,
                dino_long_side=self.dinov3_long_side,
            )
            return torch.from_numpy(patch_features), hp, wp

        self._ensure_dinov3_loaded()
        device = next(self._dinov3_model.parameters()).device
        image_pil = self._resize_image_to_dino_grid(image_pil)

        inputs = self._dinov3_image_processor(
            images=image_pil,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(device)
        outputs = self._dinov3_model(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state

        img_w, img_h = image_pil.size
        hp = img_h // self._dinov3_patch_size
        wp = img_w // self._dinov3_patch_size
        expected = hp * wp
        special_tokens = hidden.shape[1] - expected
        if special_tokens < 0:
            raise ValueError(
                f"DINO token mismatch: total={hidden.shape[1]}, expected patches={expected}"
            )
        patch_tokens = hidden[:, special_tokens:, :]
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"DINO patch token mismatch: expected={expected}, actual={patch_tokens.shape[1]}"
            )

        patch_features = F.normalize(patch_tokens[0], dim=-1).float().cpu()
        return patch_features, hp, wp

    def _normalize_zero_one(self, x: torch.Tensor, eps: float = None) -> torch.Tensor:
        eps = self.dc_eps if eps is None else eps
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + eps)

    def _build_dino_affinity(self, patch_features: torch.Tensor, eps: float = None) -> torch.Tensor:
        eps = self.dc_eps if eps is None else eps
        patch_features = F.normalize(patch_features.float(), dim=-1)
        affinity = patch_features @ patch_features.T
        affinity = torch.clamp(affinity, min=0.0)
        if self.dino_affinity_power != 1.0:
            affinity = affinity.pow(self.dino_affinity_power)
        affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + eps)
        return affinity

    def _collaborative_refine_patch_scores(
        self,
        Sp_patch: torch.Tensor,
        So_patch: torch.Tensor,
        At_patch: torch.Tensor,
        affinity: torch.Tensor,
        iterative: bool = False,
        tau_c: float = None,
        delta: float = None,
        tmax: int = None,
        eps: float = None,
    ) -> torch.Tensor:
        tau_c = self.dc_tau_c if tau_c is None else tau_c
        delta = self.refine_delta if delta is None else delta
        tmax = self.refine_tmax if tmax is None else tmax
        eps = self.dc_eps if eps is None else eps

        Sp_current = Sp_patch.reshape(-1).float()
        So_flat = So_patch.reshape(-1).float()
        At_flat = At_patch.reshape(-1).float()

        At_max = At_flat.max()
        Atilde_t = At_flat / (At_max + eps)

        num_steps = tmax if iterative else 1
        for step_idx in range(num_steps):
            Shat_p = self._normalize_zero_one(Sp_current, eps=eps)
            C = torch.clamp(Shat_p - tau_c, min=0.0)
            Rc = affinity @ C
            Rd = torch.clamp(So_flat - Sp_current, min=0.0)
            Rbar_d = affinity @ Rd

            P_raw = Rc * Atilde_t
            P = self._normalize_zero_one(P_raw, eps=eps)

            D_raw = Rbar_d * (1.0 - Atilde_t)
            D = self._normalize_zero_one(D_raw, eps=eps)

            U_plus = (1.0 - Shat_p) * Rc * Atilde_t * (1.0 - D)
            U_minus = Shat_p * Rbar_d * (1.0 - Atilde_t) * (1.0 - P)

            Sp_next = Sp_current + 0.5 * U_plus - 0.5 * U_minus
            if iterative:
                Shat_next = self._normalize_zero_one(Sp_next, eps=eps)
                prev_region = Shat_p > 0.5
                next_region = Shat_next > 0.5
                diff = (prev_region != next_region).float().mean()
                Sp_current = Sp_next
                if diff.item() < delta:
                    break
            else:
                Sp_current = Sp_next
                break

        return Sp_current.reshape_as(Sp_patch).to(dtype=Sp_patch.dtype)

    @torch.no_grad()
    def _compute_similarity_guidance_maps(
        self,
        image_pil: Image.Image,
        text_queries: List[str],
        part_score_maps: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if self.similarity_backend == "fg_clip":
            return self._compute_fg_clip_similarity_maps(
                image_pil=image_pil,
                text_queries=text_queries,
                target_h=target_h,
                target_w=target_w,
            )
        if self.similarity_backend == "dinov3":
            return self._compute_dinov3_similarity_maps(
                image_pil=image_pil,
                part_score_maps=part_score_maps,
                target_h=target_h,
                target_w=target_w,
            )
        raise ValueError(f"Unsupported similarity backend: {self.similarity_backend}")

    def dual_correction(
        self,
        Sp: torch.Tensor,       # [H, W]  初始 part score-map（logits 或 sigmoid 均可）
        So: torch.Tensor,       # [H, W]  初始 object score-map（logits 或 sigmoid 均可）
        At: torch.Tensor,       # [H, W]  FG-CLIP text-image 相似度图，已与 Sp 对齐
        tau_c:       float = None,
        kernel_size: int   = None,
        eps:         float = None,
    ) -> torch.Tensor:
        """
        论文 Algorithm 1 的 PyTorch 实现。

        Args:
            Sp          : part score-map，[H, W]
            So          : object (whole) score-map，[H, W]
            At          : FG-CLIP text-image 相似度图，[H, W]，值域 [-1, 1]
            tau_c       : 高置信核心阈值，默认使用 self.dc_tau_c
            kernel_size : 局部传播卷积核大小（奇数），默认 self.dc_kernel_size
            eps         : 数值稳定量，默认 self.dc_eps

        Returns:
            Sp_corr : 修正后的 part score-map，[H, W]
        """
        tau_c       = tau_c       if tau_c       is not None else self.dc_tau_c
        kernel_size = kernel_size if kernel_size is not None else self.dc_kernel_size
        eps         = eps         if eps         is not None else self.dc_eps

        # ── Step 1: 归一化 FG-CLIP 相似度图 ──────────────────────────────
        #   Atilde_t(x) = At(x) / (max(At) + eps)
        At_max   = At.max()
        Atilde_t = At / (At_max + eps)                                  # [H, W]，值域 ≈ [−1/max, 1]

        # ── Step 2: 归一化 part score-map ────────────────────────────────
        #   Shat_p(x) = (Sp(x) − min) / (max − min + eps)
        Sp_min  = Sp.min()
        Sp_max  = Sp.max()
        Shat_p  = (Sp - Sp_min) / (Sp_max - Sp_min + eps)              # [H, W]，值域 [0, 1]

        # ── Step 3: 高置信核心响应 ────────────────────────────────────────
        #   C(x) = max(Sp(x) − tau_c, 0)
        C = torch.clamp(Sp - tau_c, min=0.0)                            # [H, W]

        # ── Step 4: 局部连续性支持（卷积传播）────────────────────────────
        #   Rc = Conv(C, K)
        pad  = kernel_size // 2
        K    = torch.ones(1, 1, kernel_size, kernel_size,
                          device=Sp.device, dtype=Sp.dtype)
        Rc   = F.conv2d(
            C.unsqueeze(0).unsqueeze(0),                                # [1,1,H,W]
            K,
            padding=pad,
        ).squeeze()                                                      # [H, W]

        # ── Step 5: 补全项 Pc ─────────────────────────────────────────────
        #   Pc(x) = (1 − Shat_p(x)) * Rc(x) * Atilde_t(x)
        Pc = (1.0 - Shat_p) * Rc * Atilde_t                            # [H, W]

        # ── Step 6: 对象主导漂移残差 ─────────────────────────────────────
        #   Rd(x) = max(So(x) − Sp(x), 0)
        Rd = torch.clamp(So - Sp, min=0.0)                             # [H, W]

        # ── Step 7: 抑制项 Pd ─────────────────────────────────────────────
        #   Pd(x) = Rd(x) * (1 − Shat_p(x)) * (1 − Atilde_t(x))
        Pd = Rd * (1.0 - Shat_p) * (1.0 - Atilde_t)                   # [H, W]

        # ── Step 8: 更新 part score-map ──────────────────────────────────
        #   Sp_corr(x) = Sp(x) + Pc(x) − Pd(x)
        Sp_corr = Sp + Pc - Pd                                          # [H, W]

        return Sp_corr

    # =========================================================================
    # 批量 Dual Correction（对应 fused_masks × whole_pred_masks）
    # =========================================================================
    def apply_dual_correction(
        self,
        fused_masks:      List[torch.Tensor],   # List of [N, H, W]，part fused masks
        whole_pred_masks: List[torch.Tensor],   # List of [M, H, W]，whole object masks
        image_origins:    list,                 # List of PIL.Image 或 np.ndarray（原始图像）
        text_list:        List[str],            # List[str]，每条 part 文本描述
        offset:           torch.LongTensor,     # batch offset tensor
    ) -> List[torch.Tensor]:
        """
        对 batch 中每张图、每个 part mask 执行 Dual Correction。

        Args:
            fused_masks      : 融合后的 part masks，List[Tensor[N, H, W]]
            whole_pred_masks : whole-object masks，List[Tensor[M, H, W]]
            image_origins    : 原始图像列表（PIL.Image 或 np.ndarray）
            text_list        : 所有样本的文本列表（按 offset 对应）
            offset           : batch 边界，长度 = batch_size + 1

        Returns:
            corrected_masks : List[Tensor[N, H, W]]，修正后的 part masks
        """
        corrected_masks = []

        for batch_idx in range(len(fused_masks)):
            batch_part  = fused_masks[batch_idx]       # [N, H, W]
            batch_whole = whole_pred_masks[batch_idx]  # [M, H, W]
            N, H, W     = batch_part.shape

            # 取该 batch 对应的 whole mask（合并为单张二值均值图）
            # whole 可能有多个 token，这里取 sigmoid 均值作为 So
            So = batch_whole.sigmoid().mean(dim=0)     # [H, W]

            # 获取原始图像
            raw_img = image_origins[batch_idx]
            if isinstance(raw_img, np.ndarray):
                raw_img = Image.fromarray(raw_img.astype(np.uint8))

            # 获取该 batch 对应的文本描述
            start_idx = int(offset[batch_idx].item())
            end_idx   = int(offset[batch_idx + 1].item())
            batch_texts = text_list[start_idx:end_idx]  # List[str]，长度 = N

            if len(batch_texts) < N:
                batch_texts = batch_texts + [""] * (N - len(batch_texts))
            else:
                batch_texts = batch_texts[:N]

            if self.refine_method == "legacy":
                At_batch = self._compute_similarity_guidance_maps(
                    image_pil=raw_img,
                    text_queries=batch_texts,
                    part_score_maps=batch_part,
                    target_h=H,
                    target_w=W,
                ).to(device=batch_part.device, dtype=batch_part.dtype)
                affinity = None
                hp = wp = None
            else:
                At_batch = self._compute_fg_clip_similarity_maps(
                    image_pil=raw_img,
                    text_queries=batch_texts,
                    target_h=H,
                    target_w=W,
                ).to(device=batch_part.device, dtype=batch_part.dtype)
                patch_features, hp, wp = self._compute_dino_patch_features(raw_img)
                affinity = self._build_dino_affinity(
                    patch_features.to(device=batch_part.device, dtype=batch_part.dtype)
                )
                So_patch = F.interpolate(
                    So.unsqueeze(0).unsqueeze(0),
                    size=(hp, wp),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            corrected_batch = []
            for mask_idx in range(N):
                Sp = batch_part[mask_idx]              # [H, W]
                At = At_batch[mask_idx].to(device=Sp.device, dtype=Sp.dtype)

                # 获取当前 part 的文本查询

                # 计算 FG-CLIP 相似度图 At（与 mask 分辨率一致）

                # 执行 Dual Correction
                if self.refine_method == "legacy":
                    Sp_corr = self.dual_correction(
                        Sp=Sp,
                        So=So,
                        At=At,
                    )
                else:
                    Sp_patch = F.interpolate(
                        Sp.unsqueeze(0).unsqueeze(0),
                        size=(hp, wp),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                    At_patch = F.interpolate(
                        At.unsqueeze(0).unsqueeze(0),
                        size=(hp, wp),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)
                    Sp_corr_patch = self._collaborative_refine_patch_scores(
                        Sp_patch=Sp_patch,
                        So_patch=So_patch,
                        At_patch=At_patch,
                        affinity=affinity,
                        iterative=self.refine_method == "collab_iterative",
                    )
                    delta_patch = Sp_corr_patch - Sp_patch
                    delta_high = F.interpolate(
                        delta_patch.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(0).squeeze(0).to(dtype=Sp.dtype)
                    Sp_corr = Sp + self.refine_residual_weight * delta_high
                corrected_batch.append(Sp_corr)

            corrected_masks.append(torch.stack(corrected_batch, dim=0))  # [N, H, W]

        return corrected_masks

    def correct_masks_with_whole_predictions(
        self,
        corrected_masks: List[torch.Tensor],
        part_pred_masks: List[torch.Tensor],
        whole_pred_masks: List[torch.Tensor],
        eps: float = None,
    ) -> List[torch.Tensor]:
        eps = self.dc_eps if eps is None else eps
        if corrected_masks is None or part_pred_masks is None or whole_pred_masks is None:
            return corrected_masks

        corrected_again = []
        for batch_idx, batch_corr in enumerate(corrected_masks):
            if batch_idx >= len(whole_pred_masks) or batch_idx >= len(part_pred_masks):
                corrected_again.append(batch_corr)
                continue

            batch_part = part_pred_masks[batch_idx]
            batch_whole = whole_pred_masks[batch_idx]
            if batch_part is None or batch_part.numel() == 0 or batch_whole is None or batch_whole.numel() == 0:
                corrected_again.append(batch_corr)
                continue

            if batch_part.dim() == 2:
                batch_part = batch_part.unsqueeze(0)
            if batch_whole.dim() == 2:
                batch_whole = batch_whole.unsqueeze(0)

            if self.whole_guidance_mode == "soft_gate":
                whole_mean_gate = batch_whole.sigmoid().mean(dim=0, keepdim=True).detach()
                corr_prob = batch_corr.sigmoid()
                gated_prob = corr_prob * whole_mean_gate
                gated_prob = gated_prob.clamp(min=eps, max=1.0 - eps)
                corrected_again.append(torch.logit(gated_prob))
                continue

            if self.whole_guidance_mode != "contrast_part":
                raise ValueError(f"Unsupported whole guidance mode: {self.whole_guidance_mode}")

            corrected_batch = []
            for mask_idx, corr_mask in enumerate(batch_corr):
                part_mask = batch_part[min(mask_idx, batch_part.shape[0] - 1)]
                whole_mask = batch_whole[0] if batch_whole.shape[0] == 1 else batch_whole[min(mask_idx, batch_whole.shape[0] - 1)]

                corr_prob = corr_mask.sigmoid()
                corr_binary = corr_mask > 0
                part_binary = (part_mask > 0).detach()
                whole_binary = (whole_mask > 0).detach()

                corr_total = float(corr_binary.sum().item())
                corr_overlap_ratio = 0.0
                if corr_total > 0:
                    corr_overlap_ratio = float((corr_binary & whole_binary).sum().item()) / corr_total

                score = corr_prob
                if corr_overlap_ratio < self.whole_overlap_thresh:
                    score = score * whole_binary.to(dtype=score.dtype)

                part_total = float(part_binary.sum().item())
                part_overlap_ratio = 0.0
                if part_total > 0:
                    part_overlap_ratio = float((part_binary & whole_binary).sum().item()) / part_total
                if part_overlap_ratio >= self.whole_overlap_thresh:
                    score = score * part_binary.to(dtype=score.dtype)

                value_max = score.max()
                value_min = score.min()
                
                needs_contrast_enhance = (
                    value_max.item() <= self.whole_value_high_thresh
                    or value_min.item() >= self.whole_value_low_thresh
                )
                # needs_contrast_enhance = False
                if needs_contrast_enhance:
                    score = torch.clamp(
                        (score - value_min) / (value_max - value_min + eps),
                        min=0.0,
                        max=1.0,
                    )

                score = score.clamp(min=eps, max=1.0 - eps)
                corrected_batch.append(torch.logit(score))

            corrected_again.append(torch.stack(corrected_batch, dim=0))

        return corrected_again


    # =========================================================================
    # SAM visual embedding
    # =========================================================================
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            early_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings, early_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
                early_embeddings_list.append(early_embeddings[0].permute(0, 3, 1, 2))
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
            early_embeddings = torch.cat(early_embeddings_list, 0)
        return image_embeddings, early_embeddings

    def apply_mask_to_image1(self, images_raw, whole_masks, strategy="blur",
                              save_debug_images=False, save_dir="./debug"):
        """根据 whole_mask 处理原始图像，然后重新进行 CLIP 预处理。"""
        import numpy as np
        from PIL import Image
        import cv2
        import os
        from datetime import datetime

        if save_debug_images:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        masked_images_clip = []

        for i, (image_raw, mask) in enumerate(zip(images_raw, whole_masks)):
            if isinstance(image_raw, Image.Image):
                image_np = np.array(image_raw)
            else:
                image_np = image_raw

            if isinstance(mask, torch.Tensor):
                if len(mask.shape) == 3:
                    mask_combined = mask.max(dim=0)[0]
                else:
                    mask_combined = mask
                mask_np = mask_combined.cpu().numpy()
            else:
                mask_np = mask

            if mask_np.shape != image_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))

            mask_binary = (mask_np > 0.5).astype(np.float32)

            if strategy == "zero":
                if len(image_np.shape) == 3:
                    masked_image = image_np * mask_binary[:, :, np.newaxis]
                else:
                    masked_image = image_np * mask_binary
            elif strategy == "mean":
                if len(image_np.shape) == 3:
                    mean_val = image_np.mean(axis=(0, 1))
                    masked_image = (image_np * mask_binary[:, :, np.newaxis] +
                                    mean_val * (1 - mask_binary[:, :, np.newaxis]))
                else:
                    mean_val = image_np.mean()
                    masked_image = image_np * mask_binary + mean_val * (1 - mask_binary)
            elif strategy == "blur":
                blurred = cv2.GaussianBlur(image_np, (21, 21), 0)
                if len(image_np.shape) == 3:
                    masked_image = (image_np * mask_binary[:, :, np.newaxis] +
                                    blurred * (1 - mask_binary[:, :, np.newaxis]))
                else:
                    masked_image = image_np * mask_binary + blurred * (1 - mask_binary)
            elif strategy == "white":
                if len(image_np.shape) == 3:
                    white_val = np.array([255, 255, 255])
                    masked_image = (image_np * mask_binary[:, :, np.newaxis] +
                                    white_val * (1 - mask_binary[:, :, np.newaxis]))
                else:
                    masked_image = image_np * mask_binary + 255 * (1 - mask_binary)
            else:
                masked_image = image_np

            masked_image = np.clip(masked_image, 0, 255).astype(np.uint8)
            masked_pil = Image.fromarray(masked_image)

            masked_clip = self.clip_image_processor.preprocess(
                masked_pil, return_tensors="pt"
            )["pixel_values"][0]

            if hasattr(self.model, "dtype"):
                target_dtype = self.model.dtype
            else:
                target_dtype = next(self.parameters()).dtype

            masked_clip = masked_clip.to(dtype=target_dtype)
            masked_images_clip.append(masked_clip)

        return torch.stack(masked_images_clip, dim=0)

    # =========================================================================
    # forward / model_forward
    # =========================================================================
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        text_list,
        ans_list,
        image_paths,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        image_origins: np.array,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        part_input_ids: torch.LongTensor = None,
        part_labels: torch.LongTensor = None,
        part_attention_masks: torch.LongTensor = None,
        whole_input_ids: torch.LongTensor = None,
        whole_labels: torch.LongTensor = None,
        whole_attention_masks: torch.LongTensor = None,
        sent_idlist=[],
        **kwargs,
    ):
        image_embeddings, early_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # =====================================================================
        # Stage 1: Whole Segmentation
        # =====================================================================
        if whole_input_ids is not None:
            whole_seg_token_mask = whole_input_ids[:, 1:] == self.seg_token_idx
            whole_seg_token_mask = torch.cat(
                [whole_seg_token_mask,
                 torch.zeros((whole_seg_token_mask.shape[0], 1)).bool().cuda()],
                dim=1,
            )
            whole_seg_token_mask = torch.cat(
                [torch.zeros((whole_seg_token_mask.shape[0], 255)).bool().cuda(),
                 whole_seg_token_mask],
                dim=1,
            )

            if inference:
                n_batch = 1
                length  = whole_input_ids.shape[0]
                assert images_clip.shape[0] == 1
                whole_images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

                whole_output_hidden_states = []
                for i in range(n_batch):
                    start_i, end_i = i * length, min((i + 1) * length, whole_input_ids.shape[0])
                    output_i = super().forward(
                        images=whole_images_clip_extend[: end_i - start_i],
                        attention_mask=whole_attention_masks[start_i:end_i],
                        input_ids=whole_input_ids[start_i:end_i],
                        output_hidden_states=True,
                    )
                    pred_ids = torch.argmax(output_i.logits, dim=-1)
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        "/data_16T/tc/huliwen/MMR-main/M2SA-7B",
                        cache_dir=None,
                        model_max_length=2048,
                        padding_side="right",
                        use_fast=False,
                    )
                    texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                    whole_output_hidden_states.append(output_i.hidden_states)
                    torch.cuda.empty_cache()

                whole_output_hidden_states_list  = []
                whole_output_hidden_states_level = torch.cat(whole_output_hidden_states, dim=0)
                whole_output_hidden_states_list.append(whole_output_hidden_states_level)
                whole_output_hidden_states = whole_output_hidden_states_list
                whole_output = None

            else:
                whole_images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    images_clip_i = (
                        images_clip[i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    whole_images_clip_list.append(images_clip_i)
                whole_images_clip = torch.cat(whole_images_clip_list, dim=0)

                whole_output = super().forward(
                    images=whole_images_clip,
                    attention_mask=whole_attention_masks,
                    input_ids=whole_input_ids,
                    labels=whole_labels,
                    output_hidden_states=True,
                )
                whole_output_hidden_states = whole_output.hidden_states

            whole_hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            whole_hidden_states.append(
                self.model.text_hidden_fcs[0](whole_output_hidden_states[-1])
            )
            whole_last_hidden_state = torch.stack(whole_hidden_states, dim=-1).sum(dim=-1)

            whole_pred_embeddings   = whole_last_hidden_state[whole_seg_token_mask]
            whole_seg_token_counts  = whole_seg_token_mask.int().sum(-1)
            whole_seg_token_offset  = whole_seg_token_counts.cumsum(-1)
            whole_seg_token_offset  = torch.cat(
                [torch.zeros(1).long().cuda(), whole_seg_token_offset], dim=0
            )
            whole_seg_token_offset  = whole_seg_token_offset[offset]

            whole_pred_embeddings_ = []
            for i in range(len(whole_seg_token_offset) - 1):
                start_i, end_i = whole_seg_token_offset[i], whole_seg_token_offset[i + 1]
                whole_pred_embeddings_.append(whole_pred_embeddings[start_i:end_i])
            whole_pred_embeddings = whole_pred_embeddings_

            multimask_output = False
            whole_pred_masks = []
            for i in range(len(whole_pred_embeddings)):
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=whole_pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(whole_pred_embeddings[i].dtype)

                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=(
                        image_embeddings[i].unsqueeze(0),
                        early_embeddings[i].unsqueeze(0),
                    ),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                whole_pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                whole_pred_masks.append(whole_pred_mask[:, 0])

        else:
            whole_pred_masks = None

        # =====================================================================
        # Stage 2: Part Segmentation
        # =====================================================================
        if part_input_ids is not None and whole_pred_masks is not None:

            masked_images_clip = self.apply_mask_to_image1(
                image_origins,
                [mask.max(dim=0)[0] for mask in whole_pred_masks],
                strategy="blur",
            )

            part_seg_token_mask = part_input_ids[:, 1:] == self.seg_token_idx
            part_seg_token_mask = torch.cat(
                [part_seg_token_mask,
                 torch.zeros((part_seg_token_mask.shape[0], 1)).bool().cuda()],
                dim=1,
            )
            part_seg_token_mask = torch.cat(
                [torch.zeros((part_seg_token_mask.shape[0], 255)).bool().cuda(),
                 part_seg_token_mask],
                dim=1,
            )

            if inference:
                n_batch = 1
                length  = part_input_ids.shape[0]
                part_masked_images_clip_extend = masked_images_clip.expand(
                    length, -1, -1, -1
                ).contiguous()

                part_output_hidden_states = []
                for i in range(n_batch):
                    start_i, end_i = i * length, min((i + 1) * length, part_input_ids.shape[0])
                    output_i = super().forward(
                        images=part_masked_images_clip_extend[: end_i - start_i],
                        attention_mask=part_attention_masks[start_i:end_i],
                        input_ids=part_input_ids[start_i:end_i],
                        output_hidden_states=True,
                    )
                    part_output_hidden_states.append(output_i.hidden_states)
                    torch.cuda.empty_cache()

                part_output_hidden_states_list  = []
                part_output_hidden_states_level = torch.cat(part_output_hidden_states, dim=0)
                part_output_hidden_states_list.append(part_output_hidden_states_level)
                part_output_hidden_states = part_output_hidden_states_list
                part_output = None

            else:
                part_images_clip_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    images_clip_i = (
                        masked_images_clip[i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    part_images_clip_list.append(images_clip_i)
                part_masked_images_clip = torch.cat(part_images_clip_list, dim=0)

                part_output = super().forward(
                    images=part_masked_images_clip,
                    attention_mask=part_attention_masks,
                    input_ids=part_input_ids,
                    labels=part_labels,
                    output_hidden_states=True,
                )
                part_output_hidden_states = part_output.hidden_states

            part_hidden_states = []
            assert len(self.model.text_hidden_fcs) == 1
            part_hidden_states.append(
                self.model.text_hidden_fcs[0](part_output_hidden_states[-1])
            )
            part_last_hidden_state = torch.stack(part_hidden_states, dim=-1).sum(dim=-1)

            part_pred_embeddings  = part_last_hidden_state[part_seg_token_mask]
            part_seg_token_counts = part_seg_token_mask.int().sum(-1)
            part_seg_token_offset = part_seg_token_counts.cumsum(-1)
            part_seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), part_seg_token_offset], dim=0
            )
            part_seg_token_offset = part_seg_token_offset[offset]

            part_pred_embeddings_ = []
            for i in range(len(part_seg_token_offset) - 1):
                start_i, end_i = part_seg_token_offset[i], part_seg_token_offset[i + 1]
                part_pred_embeddings_.append(part_pred_embeddings[start_i:end_i])
            part_pred_embeddings = part_pred_embeddings_

            # ── part SAM decode ───────────────────────────────────────────────
            final_pred_masks = []
            for i in range(len(part_pred_embeddings)):
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=part_pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(part_pred_embeddings[i].dtype)

                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=(
                        image_embeddings[i].unsqueeze(0),
                        early_embeddings[i].unsqueeze(0),
                    ),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

                final_pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                final_pred_masks.append(final_pred_mask[:, 0])

            model_output = part_output

            # ── part / whole mask 融合（与原逻辑一致）────────────────────────
            fused_masks = []
            mask_selection_stats = {"pred_mask_used": 0, "whole_mask_used": 0}

            for batch_idx in range(len(final_pred_masks)):
                batch_part_masks  = final_pred_masks[batch_idx]   # [N, H, W]
                batch_whole_masks = whole_pred_masks[batch_idx]   # [M, H, W]

                start_idx    = offset[batch_idx]
                end_idx      = offset[batch_idx + 1]
                batch_ans_list = ans_list[start_idx:end_idx]

                fused_batch_masks = []
                for mask_idx, (mask_p, ans) in enumerate(zip(batch_part_masks, batch_ans_list)):
                    mask_w = (
                        batch_whole_masks[0]
                        if len(batch_whole_masks.shape) == 2
                        else batch_whole_masks[mask_idx]
                    )

                    mask_p_binary = (mask_p > 0).int().cpu().numpy()
                    mask_w_binary = (mask_w > 0).int().cpu().numpy()
                    part_total    = mask_p_binary.sum()

                    if part_total == 0:
                        fused_batch_masks.append(mask_w)
                        mask_selection_stats["whole_mask_used"] += 1
                    else:
                        part_in_whole = ((mask_p_binary == 1) & (mask_w_binary == 1)).sum()
                        ratio = part_in_whole / part_total

                        if ratio >= 0.8:
                            constrained_mask = mask_p_binary & mask_w_binary
                            selected_mask = torch.from_numpy(constrained_mask).to(
                                device=mask_p.device, dtype=mask_p.dtype
                            )
                            mask_selection_stats["pred_mask_used"] += 1
                        else:
                            selected_mask = mask_w
                            mask_selection_stats["whole_mask_used"] += 1

                        fused_batch_masks.append(selected_mask)

                fused_masks.append(torch.stack(fused_batch_masks, dim=0))

            # =================================================================
            # Dual Correction
            #   Sp = fused_masks（part score-map）
            #   So = whole_pred_masks（object score-map）
            #   At = FG-CLIP text-image similarity map（实时计算）
            # 输出 corrected_masks 直接替代 fused_masks 用于损失计算
            # =================================================================
            corrected_masks = self.apply_dual_correction(
                fused_masks=fused_masks,
                whole_pred_masks=whole_pred_masks,
                image_origins=image_origins,
                text_list=text_list,
                offset=offset,
            )
            corrected_masks = self.correct_masks_with_whole_predictions(
                corrected_masks=corrected_masks,
                part_pred_masks=fused_masks,
                whole_pred_masks=whole_pred_masks,
            )

        # # =====================================================================
        # # Stage 3: 主干 (input_ids) 分支，用于语言建模损失
        # # =====================================================================
        # seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        # seg_token_mask = torch.cat(
        #     [seg_token_mask,
        #      torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda()],
        #     dim=1,
        # )
        # seg_token_mask = torch.cat(
        #     [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
        #     dim=1,
        # )

        # if inference:
        #     n_batch = 1
        #     length  = input_ids.shape[0]
        #     assert images_clip.shape[0] == 1
        #     images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

        #     output_hidden_states = []
        #     for i in range(n_batch):
        #         start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
        #         output_i = super().forward(
        #             images=images_clip_extend[: end_i - start_i],
        #             attention_mask=attention_masks[start_i:end_i],
        #             input_ids=input_ids[start_i:end_i],
        #             output_hidden_states=True,
        #         )
        #         output_hidden_states.append(output_i.hidden_states)
        #         torch.cuda.empty_cache()

        #     output_hidden_states_list  = []
        #     output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
        #     output_hidden_states_list.append(output_hidden_states_level)
        #     output_hidden_states = output_hidden_states_list
        #     output = None

        # else:
        #     images_clip_list = []
        #     for i in range(len(offset) - 1):
        #         start_i, end_i = offset[i], offset[i + 1]
        #         images_clip_i = (
        #             images_clip[i]
        #             .unsqueeze(0)
        #             .expand(end_i - start_i, -1, -1, -1)
        #             .contiguous()
        #         )
        #         images_clip_list.append(images_clip_i)
        #     images_clip = torch.cat(images_clip_list, dim=0)

        #     output = super().forward(
        #         images=images_clip,
        #         attention_mask=attention_masks,
        #         input_ids=input_ids,
        #         labels=labels,
        #         output_hidden_states=True,
        #     )
        #     output_hidden_states = output.hidden_states

        # hidden_states = []
        # assert len(self.model.text_hidden_fcs) == 1
        # hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        # last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        # pred_embeddings  = last_hidden_state[seg_token_mask]
        # seg_token_counts = seg_token_mask.int().sum(-1)
        # seg_token_offset = seg_token_counts.cumsum(-1)
        # seg_token_offset = torch.cat(
        #     [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        # )
        # seg_token_offset = seg_token_offset[offset]

        # pred_embeddings_ = []
        # for i in range(len(seg_token_offset) - 1):
        #     start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
        #     pred_embeddings_.append(pred_embeddings[start_i:end_i])
        # pred_embeddings = pred_embeddings_

        # multimask_output  = False
        # pred_masks_origin = []
        # for i in range(len(pred_embeddings)):
        #     sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
        #         points=None,
        #         boxes=None,
        #         masks=None,
        #         text_embeds=pred_embeddings[i].unsqueeze(1),
        #     )
        #     sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

        #     low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
        #         image_embeddings=(
        #             image_embeddings[i].unsqueeze(0),
        #             early_embeddings[i].unsqueeze(0),
        #         ),
        #         image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=multimask_output,
        #     )

        #     pred_mask_origin = self.model.visual_model.postprocess_masks(
        #         low_res_masks,
        #         input_size=resize_list[i],
        #         original_size=label_list[i].shape,
        #     )
        #     pred_masks_origin.append(pred_mask_origin[:, 0])

        #实测删
        pred_masks_origin = fused_masks
        
        
        
        
        gt_masks = masks_list

        numbers = []
        for path in image_paths:
            name  = Path(path).stem
            match = re.search(r"0*(\d+)$", name)
            if match:
                numbers.append(int(match.group(1)))

        # ── Inference 返回 ────────────────────────────────────────────────────
        if inference:
            return {
                "text_list":          text_list,
                "ans_list":           ans_list,
                "pred_masks_origin":  pred_masks_origin,
                "fused_masks":        fused_masks,
                "corrected_masks":    corrected_masks,   # Dual Correction 输出
                "gt_masks":           gt_masks,
                "whole_pred_masks":   whole_pred_masks,
                "part_pred_masks":    final_pred_masks,
                "image_paths":        numbers,
                "sent_ids":           sent_idlist,
            }

        # ── Training 损失计算（使用 corrected_masks）─────────────────────────
        if model_output is not None:
            ce_loss = model_output.loss
        else:
            ce_loss = torch.tensor(0.0, device=images.device)

        ce_loss      = ce_loss * self.ce_loss_weight
        mask_bce_loss  = 0
        mask_dice_loss = 0
        num_masks      = 0

        for batch_idx in range(len(corrected_masks)):
            gt_mask   = gt_masks[batch_idx]
            pred_mask = corrected_masks[batch_idx]   # ← Dual Correction 结果

            assert gt_mask.shape[0] == pred_mask.shape[0], (
                f"gt_mask.shape: {gt_mask.shape}, pred_mask.shape: {pred_mask.shape}"
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss  = self.bce_loss_weight  * mask_bce_loss  / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss      = mask_bce_loss + mask_dice_loss

        whole_ce_loss = 0
        if whole_output is not None:
            whole_ce_loss = whole_output.loss * self.ce_loss_weight

        total_loss = ce_loss + mask_loss + whole_ce_loss

        return {
            "loss":           total_loss,
            "ce_loss":        ce_loss,
            "mask_bce_loss":  mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss":      mask_loss,
            "whole_ce_loss":  whole_ce_loss,
        }

    # =========================================================================
    # evaluate（保持原接口不变）
    # =========================================================================
    def evaluate(
        self,
        images_clip,
        images,
        part_input_ids,
        whole_input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=2048,
        tokenizer=None,
        local_rank=0,
    ):
        with torch.no_grad():
            part_outputs = self.generate(
                images=images_clip,
                input_ids=part_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            whole_outputs = self.generate(
                images=images_clip,
                input_ids=whole_input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            part_output_hidden_states  = part_outputs.hidden_states[-1]
            part_output_ids            = part_outputs.sequences
            whole_output_hidden_states = whole_outputs.hidden_states[-1]
            whole_output_ids           = whole_outputs.sequences

            part_seg_token_mask  = part_output_ids[:, 1:]  == self.part_seg_token_idx
            whole_seg_token_mask = whole_output_ids[:, 1:] == self.whole_seg_token_idx

            part_seg_token_mask = torch.cat(
                [torch.zeros((part_seg_token_mask.shape[0], 255)).bool().cuda(),
                 part_seg_token_mask], dim=1,
            )
            whole_seg_token_mask = torch.cat(
                [torch.zeros((whole_seg_token_mask.shape[0], 255)).bool().cuda(),
                 whole_seg_token_mask], dim=1,
            )

            part_hidden_states  = []
            whole_hidden_states = []
            assert len(self.model.part_text_hidden_fcs)  == 1
            assert len(self.model.whole_text_hidden_fcs) == 1

            part_hidden_states.append(
                self.model.part_text_hidden_fcs[0](part_output_hidden_states)
            )
            whole_hidden_states.append(
                self.model.whole_text_hidden_fcs[0](whole_output_hidden_states)
            )

            part_last_hidden_state  = torch.stack(part_hidden_states,  dim=-1).sum(dim=-1)
            whole_last_hidden_state = torch.stack(whole_hidden_states, dim=-1).sum(dim=-1)

            part_pred_embeddings  = part_last_hidden_state[part_seg_token_mask]
            whole_pred_embeddings = whole_last_hidden_state[whole_seg_token_mask]

            part_seg_token_counts  = part_seg_token_mask.int().sum(-1)
            part_seg_token_offset  = part_seg_token_counts.cumsum(-1)
            part_seg_token_offset  = torch.cat(
                [torch.zeros(1).long().cuda(), part_seg_token_offset], dim=0
            )

            part_pred_embeddings_ = []
            for i in range(len(part_seg_token_offset) - 1):
                start_i, end_i = part_seg_token_offset[i], part_seg_token_offset[i + 1]
                part_pred_embeddings_.append(part_pred_embeddings[start_i:end_i])
            part_pred_embeddings = part_pred_embeddings_

            whole_seg_token_counts = whole_seg_token_mask.int().sum(-1)
            whole_seg_token_offset = whole_seg_token_counts.cumsum(-1)
            whole_seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), whole_seg_token_offset], dim=0
            )

            whole_pred_embeddings_ = []
            for i in range(len(whole_seg_token_offset) - 1):
                start_i, end_i = whole_seg_token_offset[i], whole_seg_token_offset[i + 1]
                whole_pred_embeddings_.append(whole_pred_embeddings[start_i:end_i])
            whole_pred_embeddings = whole_pred_embeddings_

            image_embeddings, early_embeddings = self.get_visual_embs(images)

            multimask_output = False
            part_pred_masks  = []
            whole_pred_masks = []
            final_pred_masks = []

            for i in range(len(part_pred_embeddings)):
                if len(part_pred_embeddings[i]) > 0:
                    part_sparse_embeddings, part_dense_embeddings = \
                        self.model.visual_model.prompt_encoder(
                            points=None, boxes=None, masks=None,
                            text_embeds=part_pred_embeddings[i].unsqueeze(1),
                        )
                    part_sparse_embeddings = part_sparse_embeddings.to(
                        part_pred_embeddings[i].dtype
                    )
                    part_low_res_masks, _ = self.model.visual_model.mask_decoder(
                        image_embeddings=(
                            image_embeddings[i].unsqueeze(0),
                            early_embeddings[i].unsqueeze(0),
                        ),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=part_sparse_embeddings,
                        dense_prompt_embeddings=part_dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    part_pred_mask = self.model.visual_model.postprocess_masks(
                        part_low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                    part_pred_masks.append(part_pred_mask[:, 0])
                else:
                    part_pred_masks.append(torch.empty(0, *original_size_list[i]).cuda())

                if len(whole_pred_embeddings[i]) > 0:
                    whole_sparse_embeddings, whole_dense_embeddings = \
                        self.model.visual_model.prompt_encoder(
                            points=None, boxes=None, masks=None,
                            text_embeds=whole_pred_embeddings[i].unsqueeze(1),
                        )
                    whole_sparse_embeddings = whole_sparse_embeddings.to(
                        whole_pred_embeddings[i].dtype
                    )
                    whole_low_res_masks, _ = self.model.visual_model.mask_decoder(
                        image_embeddings=(
                            image_embeddings[i].unsqueeze(0),
                            early_embeddings[i].unsqueeze(0),
                        ),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=whole_sparse_embeddings,
                        dense_prompt_embeddings=whole_dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    whole_pred_mask = self.model.visual_model.postprocess_masks(
                        whole_low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                    whole_pred_masks.append(whole_pred_mask[:, 0])
                else:
                    whole_pred_masks.append(torch.empty(0, *original_size_list[i]).cuda())

                part_mask  = part_pred_masks[i]
                whole_mask = whole_pred_masks[i]

                if len(part_mask) > 0 and len(whole_mask) > 0:
                    intersection_mask = part_mask * whole_mask
                    final_pred_masks.append(intersection_mask)
                elif len(part_mask) > 0:
                    final_pred_masks.append(torch.sigmoid(part_mask))
                elif len(whole_mask) > 0:
                    final_pred_masks.append(torch.sigmoid(whole_mask))
                else:
                    final_pred_masks.append(torch.empty(0, *original_size_list[i]).cuda())

        return (whole_output_ids, part_output_ids), final_pred_masks
