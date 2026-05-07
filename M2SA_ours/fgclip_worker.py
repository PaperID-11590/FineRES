"""
FG-CLIP 独立 worker 进程
使用 Python 3.10 + transformers 5.x 环境运行
通过 multiprocessing Queue 接收请求，返回相似度图
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Queue
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM


def worker_main(request_queue: Queue, response_queue: Queue, fg_clip_root: str):
    """
    worker 主循环，在独立进程里跑，持续监听请求。

    请求格式（放入 request_queue）:
        {
            "image_np": np.ndarray,   # HWC uint8
            "text":     str,          # part 文本描述
            "target_h": int,
            "target_w": int,
            "req_id":   int,          # 请求 ID，原样返回
        }
        或发送字符串 "STOP" 来终止 worker。

    响应格式（放入 response_queue）:
        {
            "req_id":   int,
            "sim_map":  np.ndarray,   # float32, shape [target_h, target_w]
        }
    """
    print(f"[FGCLIPWorker] Loading model from {fg_clip_root} ...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = (
        AutoModelForCausalLM.from_pretrained(fg_clip_root, trust_remote_code=True)
        .to(device)
        .eval()
    )
    image_processor = AutoImageProcessor.from_pretrained(fg_clip_root)
    tokenizer       = AutoTokenizer.from_pretrained(fg_clip_root)
    eps = 1e-6

    print("[FGCLIPWorker] Ready.", flush=True)

    while True:
        req = request_queue.get()

        # 收到停止信号
        if req == "STOP":
            print("[FGCLIPWorker] Stopping.", flush=True)
            break

        image_np = req["image_np"]          # HWC uint8
        text     = req["text"]
        target_h = req["target_h"]
        target_w = req["target_w"]
        req_id   = req["req_id"]

        try:
            image_pil = Image.fromarray(image_np)

            # ── 短边缩放到 2048 ──────────────────────────────────────────
            w, h   = image_pil.size
            short  = min(w, h)
            if short < 2048:
                scale     = 2048 / short
                image_pil = image_pil.resize(
                    (int(w * scale), int(h * scale)), Image.BILINEAR
                )

            # ── FG-CLIP 前向 ─────────────────────────────────────────────
            with torch.no_grad():
                image_input = image_processor(
                    images=image_pil, max_num_patches=4096, return_tensors="pt"
                ).to(device)

                dense = model.get_image_dense_feature(**image_input)  # [1, N, D]

                spatial = image_input["spatial_shapes"][0]
                feat_h  = int(spatial[0].item())
                feat_w  = int(spatial[1].item())
                n_tok   = feat_h * feat_w

                dense = dense[0, :n_tok]                              # [N, D]
                dense = dense / (dense.norm(p=2, dim=-1, keepdim=True) + eps)

                caption_input = tokenizer(
                    [text.lower()],
                    padding="max_length", max_length=64,
                    truncation=True, return_tensors="pt"
                ).to(device)
                text_feat = model.get_text_features(**caption_input, walk_type="box")
                text_feat = text_feat / (text_feat.norm(p=2, dim=-1, keepdim=True) + eps)

                sim = (dense @ text_feat.T).squeeze(-1)               # [N]
                sim_map = sim.reshape(1, 1, feat_h, feat_w)

                sim_map_up = F.interpolate(
                    sim_map, size=(target_h, target_w),
                    mode="bilinear", align_corners=False
                ).squeeze().cpu().float().numpy()                      # [H, W]

            response_queue.put({"req_id": req_id, "sim_map": sim_map_up})

        except Exception as e:
            print(f"[FGCLIPWorker] Error on req {req_id}: {e}", flush=True)
            # 出错时返回全零图，不让主进程卡死
            response_queue.put({
                "req_id": req_id,
                "sim_map": np.zeros((target_h, target_w), dtype=np.float32),
            })


if __name__ == "__main__":
    # 直接运行时的入口：从命令行读队列端口（由主进程启动时传入）
    # 实际使用见 FGCLIPClient
    pass