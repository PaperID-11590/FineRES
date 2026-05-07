"""
SAM缓存加载模块
在推理时直接加载预处理好的SAM masks，无需重新调用SAM
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from pycocotools import mask as mask_utils


class SAMMaskLoader:
    """SAM分割结果加载器 - RLE格式"""
    
    def __init__(self, cache_dir="./sam_cache_refcocom"):
        """
        Args:
            cache_dir: 缓存目录路径（与预处理时使用的目录相同）
        """
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory does not exist: {cache_dir}")
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _get_cache_path(self, image_name):
        """根据图像名称获取缓存路径"""
        base_name = Path(image_name).stem
        subdir = self.cache_dir / base_name[:4]
        return subdir / f"{base_name}.json"
    
    def load_masks(self, image_path):
        """
        从缓存加载SAM masks
        
        Args:
            image_path: 图像路径（可以是完整路径或文件名）
            
        Returns:
            sam_masks: list of dict，每个dict包含:
                - 'segmentation': np.ndarray, shape [H, W], dtype bool
                - 'area': int
                - 'bbox': list of 4 ints [x, y, w, h]
            如果未找到缓存，返回None
        """
        # 提取文件名
        image_name = os.path.basename(image_path)
        cache_path = self._get_cache_path(image_name)
        
        # 检查缓存是否存在
        if not cache_path.exists():
            self.cache_misses += 1
            return None
        
        # 加载缓存
        try:
            with open(cache_path, 'r') as f:
                rle_masks = json.load(f)
            
            sam_masks = []
            for m in rle_masks:
                rle = m['rle']
                # 将string转回bytes
                rle['counts'] = rle['counts'].encode('utf-8')
                seg = mask_utils.decode(rle).astype(bool)
                
                sam_masks.append({
                    'segmentation': seg,
                    'area': m['area'],
                    'bbox': m['bbox'],
                })
            
            self.cache_hits += 1
            return sam_masks
            
        except Exception as e:
            print(f"Error loading cache for {image_name}: {e}")
            self.cache_misses += 1
            return None
    
    def has_cache(self, image_path):
        """检查是否有缓存"""
        image_name = os.path.basename(image_path)
        cache_path = self._get_cache_path(image_name)
        return cache_path.exists()
    
    def get_statistics(self):
        """获取缓存使用统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_queries': total,
            'hit_rate': hit_rate
        }
    
    def print_statistics(self):
        """打印缓存使用统计"""
        stats = self.get_statistics()
        print(f"\nSAM Cache Statistics:")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Hit rate: {stats['hit_rate']*100:.2f}%")
