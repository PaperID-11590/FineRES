import atexit
import subprocess
import time
import uuid
from multiprocessing import shared_memory
from multiprocessing.connection import Client
from typing import List, Optional

import numpy as np


class FGClipIPCClient:
    def __init__(
        self,
        host: str,
        port: int,
        authkey: str,
        timeout: float = 600.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.authkey = authkey.encode("utf-8")
        self.timeout = float(timeout)
        self._conn = None
        self._worker_process = None
        self._spawned_worker = False
        atexit.register(self.close)

    @property
    def address(self):
        return (self.host, self.port)

    def _connect_once(self):
        self._conn = Client(self.address, authkey=self.authkey)
        return self._conn

    def _ensure_connection(self):
        if self._conn is not None:
            return self._conn
        return self._connect_once()

    def _ping_once(self) -> bool:
        try:
            self._reset_connection()
            conn = self._connect_once()
            conn.send({"type": "ping"})
            response = conn.recv()
            return response.get("status") == "ok"
        except Exception:
            self._reset_connection()
            return False

    def _reset_connection(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _request(self, payload: dict):
        last_error = None
        for _ in range(2):
            try:
                conn = self._ensure_connection()
                conn.send(payload)
                response = conn.recv()
                if response.get("status") != "ok":
                    raise RuntimeError(response.get("message", "FG-CLIP worker failed"))
                return response
            except Exception as exc:
                last_error = exc
                self._reset_connection()
        raise RuntimeError(f"FG-CLIP IPC request failed: {last_error}") from last_error

    def _wait_until_ready(self):
        deadline = time.time() + self.timeout
        last_error = None
        while time.time() < deadline:
            try:
                if self._ping_once():
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(1.0)
        raise RuntimeError(
            f"Timed out waiting for FG-CLIP worker at {self.host}:{self.port}: {last_error}"
        ) from last_error

    def connect_or_start(
        self,
        model_root: str,
        python_executable: Optional[str] = None,
        worker_script: Optional[str] = None,
        device: Optional[str] = None,
        auto_start: bool = True,
        dinov3_root: Optional[str] = None,
    ) -> None:
        if self._ping_once():
            print(f"[FGCLIP-IPC] Reusing worker at {self.host}:{self.port}")
            return
        if not auto_start:
            raise RuntimeError(
                f"FG-CLIP worker is not reachable at {self.host}:{self.port} and auto start is disabled."
            )

        if not python_executable:
            raise ValueError(
                "fg_clip_python is required when fg_clip_mode='ipc' and the worker is not already running."
            )
        if not worker_script:
            raise ValueError("fg_clip_worker_script is required to auto start the FG-CLIP worker.")

        command = [
            python_executable,
            worker_script,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--authkey",
            self.authkey.decode("utf-8"),
            "--fg_clip_root",
            model_root,
        ]
        if dinov3_root:
            command.extend(["--dinov3_root", dinov3_root])
        if device:
            command.extend(["--device", device])

        print(
            f"[FGCLIP-IPC] Starting worker with {python_executable} on "
            f"{self.host}:{self.port} (device={device or 'auto'})"
        )
        self._worker_process = subprocess.Popen(command)
        self._spawned_worker = True
        self._wait_until_ready()
        print(f"[FGCLIP-IPC] Worker ready at {self.host}:{self.port}")

    def compute_similarity_maps(
        self,
        image_array: np.ndarray,
        text_queries: Optional[List[str]],
        target_h: int,
        target_w: int,
        resize_short: int,
        max_num_patches: int,
        backend: str = "fg_clip",
        score_maps: Optional[np.ndarray] = None,
        dino_long_side: Optional[int] = None,
        dino_ref_topk: int = 1,
    ) -> np.ndarray:
        if image_array.dtype != np.uint8:
            raise ValueError("image_array must be uint8")
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("image_array must have shape [H, W, 3]")
        if backend == "fg_clip" and (text_queries is None or len(text_queries) == 0):
            return np.empty((0, target_h, target_w), dtype=np.float32)
        if backend == "dinov3" and score_maps is not None and score_maps.shape[0] == 0:
            return np.empty((0, target_h, target_w), dtype=np.float32)

        image_array = np.ascontiguousarray(image_array)
        if backend == "fg_clip":
            output_count = len(text_queries)
        elif backend == "dinov3":
            if score_maps is None:
                raise ValueError("score_maps are required for dinov3 backend")
            score_maps = np.ascontiguousarray(score_maps.astype(np.float32))
            output_count = int(score_maps.shape[0])
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        output_shape = (output_count, int(target_h), int(target_w))
        image_shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
        output_shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
        )
        score_shm = None
        if score_maps is not None:
            score_shm = shared_memory.SharedMemory(create=True, size=score_maps.nbytes)

        try:
            shared_image = np.ndarray(
                image_array.shape, dtype=image_array.dtype, buffer=image_shm.buf
            )
            shared_image[:] = image_array
            if score_shm is not None:
                shared_scores = np.ndarray(
                    score_maps.shape, dtype=score_maps.dtype, buffer=score_shm.buf
                )
                shared_scores[:] = score_maps

            payload = {
                "type": "compute_similarity_maps",
                "backend": backend,
                "request_id": str(uuid.uuid4()),
                "texts": [text or "" for text in (text_queries or [])],
                "target_h": int(target_h),
                "target_w": int(target_w),
                "resize_short": int(resize_short),
                "max_num_patches": int(max_num_patches),
                "dino_long_side": int(dino_long_side or resize_short),
                "dino_ref_topk": int(dino_ref_topk),
                "image": {
                    "name": image_shm.name,
                    "shape": tuple(image_array.shape),
                    "dtype": str(image_array.dtype),
                },
                "output": {
                    "name": output_shm.name,
                    "shape": output_shape,
                    "dtype": "float32",
                },
            }
            if score_shm is not None:
                payload["score_maps"] = {
                    "name": score_shm.name,
                    "shape": tuple(score_maps.shape),
                    "dtype": str(score_maps.dtype),
                }
            self._request(payload)
            output_array = np.ndarray(output_shape, dtype=np.float32, buffer=output_shm.buf)
            return output_array.copy()
        finally:
            try:
                image_shm.close()
            finally:
                image_shm.unlink()
            try:
                output_shm.close()
            finally:
                output_shm.unlink()
            if score_shm is not None:
                try:
                    score_shm.close()
                finally:
                    score_shm.unlink()

    def compute_dino_patch_features(
        self,
        image_array: np.ndarray,
        dino_long_side: int,
    ):
        if image_array.dtype != np.uint8:
            raise ValueError("image_array must be uint8")
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("image_array must have shape [H, W, 3]")

        image_array = np.ascontiguousarray(image_array)
        image_shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
        try:
            shared_image = np.ndarray(
                image_array.shape, dtype=image_array.dtype, buffer=image_shm.buf
            )
            shared_image[:] = image_array
            response = self._request(
                {
                    "type": "compute_dino_patch_features",
                    "dino_long_side": int(dino_long_side),
                    "image": {
                        "name": image_shm.name,
                        "shape": tuple(image_array.shape),
                        "dtype": str(image_array.dtype),
                    },
                }
            )
            patch_features = np.asarray(response["patch_features"], dtype=np.float32)
            return patch_features, int(response["Hp"]), int(response["Wp"])
        finally:
            try:
                image_shm.close()
            finally:
                image_shm.unlink()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send({"type": "shutdown"})
                self._conn.recv()
            except Exception:
                pass
        self._reset_connection()
        if self._worker_process is not None and self._spawned_worker:
            try:
                self._worker_process.wait(timeout=5)
            except Exception:
                self._worker_process.kill()
            self._worker_process = None
