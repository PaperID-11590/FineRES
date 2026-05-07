import argparse
import traceback
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import Listener

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer


class SimilarityWorker:
    def __init__(
        self,
        fg_clip_root: str,
        dinov3_root: str,
        device: str,
        resize_short: int = 2048,
        max_num_patches: int = 4096,
    ):
        self.fg_clip_root = fg_clip_root
        self.dinov3_root = dinov3_root
        self.device = self._resolve_device(device)
        self.resize_short = int(resize_short)
        self.max_num_patches = int(max_num_patches)
        self.eps = 1e-6
        self.fg_model = None
        self.fg_image_processor = None
        self.fg_tokenizer = None
        self.dino_model = None
        self.dino_image_processor = None
        self.dino_patch_size = 14

    def _resolve_device(self, device: str) -> str:
        if not device or device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def _ensure_fg_clip_loaded(self):
        if self.fg_model is not None:
            return
        if not self.fg_clip_root:
            raise ValueError("fg_clip_root is required for fg_clip backend")

        self.fg_model = (
            AutoModelForCausalLM.from_pretrained(
                self.fg_clip_root, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )
        for param in self.fg_model.parameters():
            param.requires_grad = False
        self.fg_image_processor = AutoImageProcessor.from_pretrained(self.fg_clip_root)
        self.fg_tokenizer = AutoTokenizer.from_pretrained(self.fg_clip_root)
        print(f"[SIM-WORKER] FG-CLIP loaded from {self.fg_clip_root}", flush=True)

    def _ensure_dinov3_loaded(self):
        if self.dino_model is not None:
            return
        if not self.dinov3_root:
            raise ValueError("dinov3_root is required for dinov3 backend")

        self.dino_image_processor = AutoImageProcessor.from_pretrained(self.dinov3_root)
        self.dino_model = AutoModel.from_pretrained(self.dinov3_root).to(self.device).eval()
        for param in self.dino_model.parameters():
            param.requires_grad = False
        if hasattr(self.dino_model.config, "patch_size"):
            self.dino_patch_size = int(self.dino_model.config.patch_size)
        print(
            f"[SIM-WORKER] DINOv3 loaded from {self.dinov3_root} "
            f"(patch_size={self.dino_patch_size})",
            flush=True,
        )

    def _resize_to_patch_grid(self, image_pil: Image.Image, long_side: int):
        patch = self.dino_patch_size
        img_rgb = image_pil.convert("RGB")
        width, height = img_rgb.size
        scale = float(long_side) / max(height, width)
        new_h = max(patch, int(round(height * scale)))
        new_w = max(patch, int(round(width * scale)))
        new_h = ((new_h + patch - 1) // patch) * patch
        new_w = ((new_w + patch - 1) // patch) * patch
        return img_rgb.resize((new_w, new_h), Image.LANCZOS)

    @torch.inference_mode()
    def extract_dino_patch_features(
        self,
        image_array: np.ndarray,
        dino_long_side: int,
    ):
        self._ensure_dinov3_loaded()

        image_pil = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
        image_pil = self._resize_to_patch_grid(image_pil, int(dino_long_side))
        inputs = self.dino_image_processor(
            images=image_pil,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        ).to(self.device)
        outputs = self.dino_model(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state

        img_w, img_h = image_pil.size
        hp = img_h // self.dino_patch_size
        wp = img_w // self.dino_patch_size
        expected = hp * wp
        total_tokens = hidden.shape[1]
        special_tokens = total_tokens - expected
        if special_tokens < 0:
            raise ValueError(
                f"DINO token mismatch: total={total_tokens}, expected patch tokens={expected}"
            )

        patch_tokens = hidden[:, special_tokens:, :]
        if patch_tokens.shape[1] != expected:
            raise ValueError(
                f"DINO patch token mismatch: expected={expected}, actual={patch_tokens.shape[1]}"
            )

        features = F.normalize(patch_tokens[0], dim=-1).float().cpu().numpy()
        return features, hp, wp

    @torch.inference_mode()
    def _compute_fg_clip_similarity_maps(
        self,
        image_array: np.ndarray,
        text_queries,
        target_h: int,
        target_w: int,
        resize_short: int,
        max_num_patches: int,
    ) -> np.ndarray:
        self._ensure_fg_clip_loaded()

        if len(text_queries) == 0:
            return np.empty((0, target_h, target_w), dtype=np.float32)

        image_pil = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
        img_w, img_h = image_pil.size
        short_edge = min(img_w, img_h)
        if short_edge < resize_short:
            scale = resize_short / short_edge
            image_pil = image_pil.resize(
                (int(img_w * scale), int(img_h * scale)), Image.BILINEAR
            )

        image_input = self.fg_image_processor(
            images=image_pil,
            max_num_patches=max_num_patches,
            return_tensors="pt",
        ).to(self.device)

        dense_feat = self.fg_model.get_image_dense_feature(**image_input)
        spatial = image_input["spatial_shapes"][0]
        feat_h = int(spatial[0].item())
        feat_w = int(spatial[1].item())
        num_tokens = feat_h * feat_w

        dense_feat = dense_feat[0, :num_tokens]
        dense_feat = dense_feat / (dense_feat.norm(p=2, dim=-1, keepdim=True) + self.eps)

        caption_input = self.fg_tokenizer(
            [text.lower() for text in text_queries],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        text_feat = self.fg_model.get_text_features(**caption_input, walk_type="box")
        text_feat = text_feat / (text_feat.norm(p=2, dim=-1, keepdim=True) + self.eps)

        sim = dense_feat @ text_feat.T
        sim_maps = sim.transpose(0, 1).reshape(len(text_queries), 1, feat_h, feat_w)
        sim_maps_up = F.interpolate(
            sim_maps,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return sim_maps_up.float().cpu().numpy()

    @torch.inference_mode()
    def _compute_dinov3_similarity_maps(
        self,
        image_array: np.ndarray,
        score_maps: np.ndarray,
        target_h: int,
        target_w: int,
        dino_long_side: int,
        dino_ref_topk: int,
    ) -> np.ndarray:
        if score_maps.shape[0] == 0:
            return np.empty((0, target_h, target_w), dtype=np.float32)
        features_np, hp, wp = self.extract_dino_patch_features(
            image_array=image_array,
            dino_long_side=dino_long_side,
        )
        features = torch.from_numpy(features_np).to(self.device)
        score_tensor = torch.from_numpy(score_maps).to(self.device, dtype=features.dtype).unsqueeze(1)
        score_tensor = F.interpolate(
            score_tensor,
            size=(hp, wp),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        flat_scores = score_tensor.reshape(score_tensor.shape[0], -1)

        topk = max(1, min(int(dino_ref_topk), flat_scores.shape[1]))
        if topk == 1:
            anchor_idx = flat_scores.argmax(dim=1)
            ref_features = features[anchor_idx]
        else:
            top_values, top_indices = torch.topk(flat_scores, k=topk, dim=1)
            top_features = features[top_indices]
            weights = torch.softmax(top_values, dim=1).unsqueeze(-1)
            ref_features = (top_features * weights).sum(dim=1)
            ref_features = F.normalize(ref_features, dim=-1)

        sim = features @ ref_features.T
        sim_maps = sim.transpose(0, 1).reshape(score_tensor.shape[0], 1, hp, wp)
        sim_maps_up = F.interpolate(
            sim_maps,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)
        return sim_maps_up.float().cpu().numpy()

    @torch.inference_mode()
    def compute_similarity_maps(
        self,
        backend: str,
        image_array: np.ndarray,
        text_queries,
        target_h: int,
        target_w: int,
        resize_short: int,
        max_num_patches: int,
        score_maps=None,
        dino_long_side: int = 756,
        dino_ref_topk: int = 1,
    ) -> np.ndarray:
        if backend == "fg_clip":
            return self._compute_fg_clip_similarity_maps(
                image_array=image_array,
                text_queries=text_queries,
                target_h=target_h,
                target_w=target_w,
                resize_short=resize_short,
                max_num_patches=max_num_patches,
            )
        if backend == "dinov3":
            if score_maps is None:
                raise ValueError("score_maps are required for dinov3 backend")
            return self._compute_dinov3_similarity_maps(
                image_array=image_array,
                score_maps=score_maps,
                target_h=target_h,
                target_w=target_w,
                dino_long_side=dino_long_side,
                dino_ref_topk=dino_ref_topk,
            )
        raise ValueError(f"Unsupported backend: {backend}")


def parse_args():
    parser = argparse.ArgumentParser(description="FG-CLIP worker")
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=29610, type=int)
    parser.add_argument("--authkey", default="m2sa_fgclip", type=str)
    parser.add_argument("--fg_clip_root", default="", type=str)
    parser.add_argument("--dinov3_root", default="", type=str)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--resize_short", default=2048, type=int)
    parser.add_argument("--max_num_patches", default=4096, type=int)
    return parser.parse_args()


def _unregister_shared_memory(shm_obj):
    try:
        resource_tracker.unregister(shm_obj._name, "shared_memory")
    except Exception:
        pass


def main():
    args = parse_args()
    worker = SimilarityWorker(
        fg_clip_root=args.fg_clip_root,
        dinov3_root=args.dinov3_root,
        device=args.device,
        resize_short=args.resize_short,
        max_num_patches=args.max_num_patches,
    )
    listener = Listener((args.host, args.port), authkey=args.authkey.encode("utf-8"))
    print(
        f"[FGCLIP-WORKER] Ready. host={args.host} port={args.port} "
        f"device={worker.device} fg_clip={args.fg_clip_root or 'disabled'} "
        f"dinov3={args.dinov3_root or 'disabled'}",
        flush=True,
    )

    while True:
        conn = listener.accept()
        try:
            while True:
                message = conn.recv()
                msg_type = message.get("type")

                if msg_type == "ping":
                    conn.send({"status": "ok"})
                    continue

                if msg_type == "shutdown":
                    conn.send({"status": "ok"})
                    listener.close()
                    return

                if msg_type not in {"compute_similarity_maps", "compute_dino_patch_features"}:
                    conn.send({"status": "error", "message": f"Unknown request type: {msg_type}"})
                    continue

                image_meta = message["image"]
                image_shm = shared_memory.SharedMemory(name=image_meta["name"])
                _unregister_shared_memory(image_shm)
                output_shm = None
                score_maps = None
                score_shm = None
                score_meta = None
                output_meta = message.get("output")
                if output_meta is not None:
                    output_shm = shared_memory.SharedMemory(name=output_meta["name"])
                    _unregister_shared_memory(output_shm)
                if "score_maps" in message:
                    score_meta = message["score_maps"]
                    score_shm = shared_memory.SharedMemory(name=score_meta["name"])
                    _unregister_shared_memory(score_shm)

                try:
                    image_array = np.ndarray(
                        tuple(image_meta["shape"]),
                        dtype=np.dtype(image_meta["dtype"]),
                        buffer=image_shm.buf,
                    ).copy()
                    if msg_type == "compute_dino_patch_features":
                        patch_features, hp, wp = worker.extract_dino_patch_features(
                            image_array=image_array,
                            dino_long_side=int(message.get("dino_long_side", 756)),
                        )
                        conn.send(
                            {
                                "status": "ok",
                                "patch_features": patch_features.astype(np.float32),
                                "Hp": int(hp),
                                "Wp": int(wp),
                            }
                        )
                        continue

                    backend = message.get("backend", "fg_clip")
                    if score_shm is not None:
                        score_maps = np.ndarray(
                            tuple(score_meta["shape"]),
                            dtype=np.dtype(score_meta["dtype"]),
                            buffer=score_shm.buf,
                        ).copy()
                    result = worker.compute_similarity_maps(
                        backend=backend,
                        image_array=image_array,
                        text_queries=message["texts"],
                        target_h=int(message["target_h"]),
                        target_w=int(message["target_w"]),
                        resize_short=int(message["resize_short"]),
                        max_num_patches=int(message["max_num_patches"]),
                        score_maps=score_maps,
                        dino_long_side=int(message.get("dino_long_side", message["resize_short"])),
                        dino_ref_topk=int(message.get("dino_ref_topk", 1)),
                    )
                    shared_output = np.ndarray(
                        tuple(output_meta["shape"]),
                        dtype=np.float32,
                        buffer=output_shm.buf,
                    )
                    shared_output[:] = result
                    conn.send({"status": "ok", "num_maps": int(result.shape[0])})
                finally:
                    image_shm.close()
                    if output_shm is not None:
                        output_shm.close()
                    if score_shm is not None:
                        score_shm.close()
        except EOFError:
            conn.close()
        except Exception:
            try:
                conn.send({"status": "error", "message": traceback.format_exc()})
            except Exception:
                pass
            conn.close()


if __name__ == "__main__":
    main()
