


# ============ 修改 utils/attention_vis.py ============
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch

def visualize_seg_attention(attention_map_data, save_path=None, show_plot=False):
    """
    可视化SEG token的attention map
    
    Args:
        attention_map_data: 单个conversation的attention数据字典
        save_path: 保存路径（可选）
        show_plot: 是否显示图像
    """
    print(attention_map_data)
    tokens = attention_map_data['text_tokens']
    
    # ============ 修改：处理BFloat16类型 ============
    attention_weights = attention_map_data['attention_weights']
    if isinstance(attention_weights, torch.Tensor):
        # 如果是BFloat16，先转float32再转numpy
        if attention_weights.dtype == torch.bfloat16:
            weights = attention_weights.float().numpy()
        else:
            weights = attention_weights.numpy()
    else:
        weights = attention_weights
    # ============================================
    
    original_text = attention_map_data['original_text']
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.5), 5))
    
    bars = ax.bar(range(len(tokens)), weights, alpha=0.7, color='steelblue')
    
    # 高亮最重要的tokens
    top_k = min(5, len(weights))
    top_indices = np.argsort(weights)[-top_k:]
    for idx in top_indices:
        bars[idx].set_color('coral')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_title(f'SEG Token Attention Map\nText: {original_text}', fontsize=14, pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved attention map to {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def visualize_target_words_attention(attention_map_data, save_path=None, show_plot=False):
    """
    可视化"what is"和"in this image"之间的words的attention map（归一化版本）
    
    Args:
        attention_map_data: 单个conversation的attention数据字典
        save_path: 保存路径（可选）
        show_plot: 是否显示图像
    """
    # 检查是否有提取的target words
    target_words = attention_map_data.get('target_words', None)
    target_weights = attention_map_data.get('target_weights', None)
    
    if target_words is None or target_weights is None:
        print(f"⚠️ No target words found between 'what is' and 'in this image'")
        return
    
    if len(target_words) == 0:
        print(f"⚠️ Empty target words list")
        return
    
    # ============ 处理权重数据类型 ============
    if isinstance(target_weights, torch.Tensor):
        if target_weights.dtype == torch.bfloat16:
            weights = target_weights.float().numpy()
        else:
            weights = target_weights.numpy()
    elif isinstance(target_weights, list):
        weights = np.array(target_weights)
    else:
        weights = target_weights
    # ========================================
    
    original_text = attention_map_data.get('original_text', 'N/A')
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(max(10, len(target_words) * 0.6), 5))
    
    bars = ax.bar(range(len(target_words)), weights, alpha=0.7, color='mediumseagreen')
    
    # 高亮最重要的tokens（top 3）
    top_k = min(3, len(weights))
    top_indices = np.argsort(weights)[-top_k:]
    for idx in top_indices:
        bars[idx].set_color('orangered')
    
    ax.set_xticks(range(len(target_words)))
    ax.set_xticklabels(target_words, rotation=45, ha='right', fontsize=9, weight='bold')
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_xlabel('Target Words', fontsize=12)
    ax.set_title(
        f'Target Words Attention (between "what is" and "in this image")\nText: {original_text}', 
        fontsize=14, 
        pad=20
    )
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在每个柱子上方显示权重值
    for i, (word, weight) in enumerate(zip(target_words, weights)):
        ax.text(i, weight, f'{weight:.3f}', 
                ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved target words attention map to {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def visualize_image_all_attentions(image_attention_maps, image_path, save_dir, stage_name=''):
    """
    可视化一张图像所有conversations的attention maps
    同时生成完整的attention map和target words的attention map
    
    Args:
        image_attention_maps: 一张图像的所有attention maps (list of dicts)
        image_path: 图像路径或ID
        save_dir: 保存目录
        stage_name: stage名称（用于文件命名，如'whole', 'part', 'origin'）
    """
    if not image_attention_maps or len(image_attention_maps) == 0:
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每个conversation创建两种可视化
    for attn_data in image_attention_maps:
        conv_idx = attn_data['conversation_idx']
        
        # 构建文件名前缀
        prefix = f"{stage_name}_" if stage_name else ""
        
        # 1. 完整的attention map
        full_save_path = save_dir / f"{prefix}full.png"
        visualize_seg_attention(attn_data, save_path=str(full_save_path))
        
        # 2. Target words的attention map（如果存在）
        if attn_data.get('target_words') is not None:
            target_save_path = save_dir / f"{prefix}target.png"
            visualize_target_words_attention(attn_data, save_path=str(target_save_path))
        else:
            print(f"⚠️ No target words for {stage_name} stage, conv {conv_idx}")
    
    print(f"✅ Saved attention maps for {stage_name} stage: {save_dir}")


def visualize_batch_attentions_by_sentence(
    whole_attention_maps, 
    part_attention_maps, 
    origin_attention_maps,
    image_paths,
    ans_list,
    sent_ids_list,
    texts_list,
    save_root_dir='attention_vis'
):
    """
    按句子组织批量可视化attention maps
    每个句子创建一个文件夹，包含6个文件：whole/part/origin的full和target
    
    注意：在验证模式下，每个batch_size=1，但可能有多个conversations
    
    Args:
        whole_attention_maps: whole stage的attention maps (list of list of dicts)
                             外层list对应batch，内层list对应该batch中的conversations
        part_attention_maps: part stage的attention maps
        origin_attention_maps: origin stage的attention maps
        image_paths: 图像路径列表
        ans_list: 答案列表
        sent_ids_list: 句子ID列表
        texts_list: 文本列表
        save_root_dir: 保存根目录
    """
    save_root_dir = Path(save_root_dir)
    save_root_dir.mkdir(parents=True, exist_ok=True)
    
    # print(f"\n{'='*80}")
    # print(f"Starting sentence-based batch visualization:")
    # print(f"  Number of batches: {len(whole_attention_maps) if whole_attention_maps else 0}")
    # print(f"  Save root directory: {save_root_dir}")
    # print(f"{'='*80}\n")
    
    # 遍历每个batch（在验证模式下，通常batch_size=1）
    for batch_idx in range(len(whole_attention_maps) if whole_attention_maps else 0):

        # 获取三个stage的attention maps（都是list of dicts，表示该batch中的多个conversations）
        whole_convs = whole_attention_maps[batch_idx] if (whole_attention_maps and batch_idx < len(whole_attention_maps)) else []
        part_convs = part_attention_maps[batch_idx] if (part_attention_maps and batch_idx < len(part_attention_maps)) else []
        origin_convs = origin_attention_maps[batch_idx] if (origin_attention_maps and batch_idx < len(origin_attention_maps)) else []
        
        # 获取该batch的conversation数量（应该都相同）
        num_convs = max(len(whole_convs), len(part_convs), len(origin_convs))
        
        # if num_convs == 0:
        #     print(f"  ⚠️ No conversations found in batch {batch_idx}")
        #     continue
        
        # print(f"  Number of conversations: {num_convs}")
        
        # 遍历每个conversation（句子）
        for conv_idx in range(num_convs):
            # 获取该conversation的元数据
            ans = ans_list[conv_idx] if conv_idx < len(ans_list) else "unknown"
            sent_id = sent_ids_list[conv_idx] if conv_idx < len(sent_ids_list) else conv_idx
            text = texts_list[conv_idx] if conv_idx < len(texts_list) else "no_text"
            image_path = image_paths[0]
            # 清理文本中的特殊字符
            text_clean = text.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_')
            text_clean = text_clean.replace('<', '_').replace('>', '_').replace('|', '_')
            text_clean = text_clean[:50]  # 限制长度
            
            # 构建句子专属文件夹
            sentence_folder_name = f"{image_path}_--{ans}--{sent_id}--{text_clean}"
            sentence_dir = save_root_dir / sentence_folder_name
            sentence_dir.mkdir(parents=True, exist_ok=True)
            
            # print(f"\n  Conversation {conv_idx + 1}/{num_convs}:")
            # print(f"    Sentence ID: {sent_id}")
            # print(f"    Answer: {ans}")
            # print(f"    Text: {text[:60]}...")
            # print(f"    Folder: {sentence_folder_name}")
            
            # 可视化三个stage
            has_data = False
            # 1. Whole stage
            if conv_idx < len(whole_convs) and whole_convs[conv_idx]:
                full_save_path = sentence_dir/"whole_full.png"
                target_save_path = sentence_dir/"whole_target.png"
                visualize_seg_attention(whole_convs[conv_idx],str(full_save_path))
                visualize_target_words_attention(whole_convs[conv_idx],str(target_save_path))
                has_data = True
            else:
                print(f"    ⚠️ No data for whole stage")
            
            # # 2. Part stage
            if conv_idx < len(part_convs) and part_convs[conv_idx]:
                full_save_path = sentence_dir/"part_full.png"
                target_save_path = sentence_dir/"part_target.png"
                visualize_seg_attention(part_convs[conv_idx],str(full_save_path))
                visualize_target_words_attention(part_convs[conv_idx],str(target_save_path))
                has_data = True
            else:
                print(f"    ⚠️ No data for part stage")
            
            # 3. Origin stage
            if conv_idx < len(origin_convs) and origin_convs[conv_idx]:
                full_save_path = sentence_dir/"origin_full.png"
                target_save_path = sentence_dir/"origin_target.png"
                visualize_seg_attention(origin_convs[conv_idx],str(full_save_path))
                visualize_target_words_attention(origin_convs[conv_idx],str(target_save_path))
                has_data = True
            else:
                print(f"    ⚠️ No data for origin stage")
            
            if has_data:
                print(f"    ✅ Completed conversation {conv_idx + 1}")
            else:
                print(f"    ⚠️ No attention data for this conversation, removing empty folder")
                # 删除空文件夹
                try:
                    sentence_dir.rmdir()
                except:
                    pass
    
    # print(f"\n{'='*80}")
    # print(f"✅ Batch visualization complete!")
    # print(f"   Results saved in: {save_root_dir}")
    # print(f"{'='*80}\n")


def visualize_batch_attentions(all_attention_maps, image_ids, save_dir):
    """
    批量可视化多张图像的attention maps（原始版本，保留向后兼容）
    
    Args:
        all_attention_maps: 所有图像的attention maps (list of list of dicts)
        image_ids: 图像ID列表
        save_dir: 保存根目录
    """
    if not all_attention_maps or len(all_attention_maps) == 0:
        print("⚠️ No attention maps to visualize")
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = len(all_attention_maps)
    total_convs = sum(len(img_maps) for img_maps in all_attention_maps)
    
    print(f"\n{'='*60}")
    print(f"Starting batch visualization:")
    print(f"  Total images: {total_images}")
    print(f"  Total conversations: {total_convs}")
    print(f"  Save directory: {save_dir}")
    print(f"{'='*60}\n")
    
    for img_idx, image_attention_maps in enumerate(all_attention_maps):
        image_id = image_ids[img_idx] if img_idx < len(image_ids) else f"img_{img_idx}"
        print(f"Processing image {img_idx + 1}/{total_images} (ID: {image_id})...")
        visualize_image_all_attentions(
            image_attention_maps, 
            image_id, 
            save_dir / str(image_id)
        )
    
    print(f"\n{'='*60}")
    print(f"✅ Batch visualization complete!")
    print(f"   Generated visualizations in: {save_dir}")
    print(f"{'='*60}\n")
