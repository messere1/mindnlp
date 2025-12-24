"""
批处理整理器
负责批量数据的整理和padding
支持自适应批处理和动态优化
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from utils.logger import get_logger


logger = get_logger(__name__)


class BatchCollator:
    """
    批处理整理器
    
    功能:
    1. 动态分组（按图像尺寸相似度）
    2. 智能 Padding（最小化计算浪费）
    3. 批次构建
    """
    
    def __init__(self, max_group_diff: float = 0.2):
        """
        初始化批处理整理器
        
        Args:
            max_group_diff: 最大尺寸差异比例（用于分组）
        """
        self.max_group_diff = max_group_diff
        logger.info(f"BatchCollator initialized with max_group_diff={max_group_diff}")
    
    def collate(
        self,
        images: List[torch.Tensor],
        transform_infos: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        整理批量数据
        
        Args:
            images: 图像 Tensor 列表 [每个为 [1, 3, H, W]]
            transform_infos: 变换信息列表
            
        Returns:
            batch_images: 批量图像 Tensor [B, 3, H, W]
            transform_infos: 变换信息列表（原样返回）
        """
        if not images:
            raise ValueError("Empty image list")
        
        # 移除单个 batch 维度并堆叠
        images = [img.squeeze(0) if img.dim() == 4 else img for img in images]
        
        # 堆叠成批量
        batch_images = torch.stack(images, dim=0)
        
        logger.debug(f"Collated batch: {len(images)} images, shape={batch_images.shape}")
        
        return batch_images, transform_infos
    
    def group_by_size(
        self,
        images: List[np.ndarray],
        transform_infos: List[Dict[str, Any]]
    ) -> List[Tuple[List[np.ndarray], List[Dict[str, Any]]]]:
        """
        按尺寸相似度分组
        
        Args:
            images: 图像数组列表
            transform_infos: 变换信息列表
            
        Returns:
            groups: 分组后的列表 [(images_group, infos_group), ...]
        """
        if not images:
            return []
        
        # 计算每个图像的尺寸（宽高比）
        sizes = []
        for info in transform_infos:
            orig_w, orig_h = info['original_size']
            aspect_ratio = orig_w / max(orig_h, 1)
            sizes.append((orig_w, orig_h, aspect_ratio))
        
        # 按宽高比排序
        sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i][2])
        
        # 分组
        groups = []
        current_group_images = []
        current_group_infos = []
        current_aspect = None
        
        for idx in sorted_indices:
            aspect_ratio = sizes[idx][2]
            
            if current_aspect is None:
                # 第一个元素
                current_aspect = aspect_ratio
                current_group_images.append(images[idx])
                current_group_infos.append(transform_infos[idx])
            else:
                # 检查是否应该新建组
                diff = abs(aspect_ratio - current_aspect) / max(current_aspect, 1)
                
                if diff <= self.max_group_diff:
                    # 加入当前组
                    current_group_images.append(images[idx])
                    current_group_infos.append(transform_infos[idx])
                else:
                    # 保存当前组，开始新组
                    groups.append((current_group_images, current_group_infos))
                    current_group_images = [images[idx]]
                    current_group_infos = [transform_infos[idx]]
                    current_aspect = aspect_ratio
        
        # 添加最后一组
        if current_group_images:
            groups.append((current_group_images, current_group_infos))
        
        logger.debug(f"Grouped {len(images)} images into {len(groups)} groups")
        return groups
    
    def smart_padding(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        智能 Padding（最小化计算浪费）
        
        Args:
            images: 图像数组列表 [每个为 [C, H, W]]
            
        Returns:
            padded_images: Padding 后的批量图像 [B, C, H, W]
            padding_info: Padding 信息
        """
        if not images:
            raise ValueError("Empty image list")
        
        # 找到批次中的最大尺寸
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        # 对齐到 32 的倍数（优化GPU计算）
        max_h = ((max_h + 31) // 32) * 32
        max_w = ((max_w + 31) // 32) * 32
        
        # 创建 Padding 后的批量数组
        batch_size = len(images)
        channels = images[0].shape[0]
        padded_images = np.zeros((batch_size, channels, max_h, max_w), dtype=np.float32)
        
        # Padding 每个图像
        for i, img in enumerate(images):
            c, h, w = img.shape
            padded_images[i, :, :h, :w] = img
        
        padding_info = {
            'max_height': max_h,
            'max_width': max_w,
            'original_shapes': [img.shape for img in images]
        }
        
        logger.debug(f"Smart padding: {batch_size} images to {max_h}x{max_w}")
        
        return padded_images, padding_info


class AdaptiveBatchCollator:
    """
    自适应批处理整理器
    
    特性:
    1. 智能分组 - 按图像尺寸相似度动态分组
    2. 自适应批大小 - 根据GPU内存动态调整
    3. 最小化填充 - 减少计算浪费
    4. 内存监控 - 防止OOM
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        size_threshold: float = 0.2,
        target_gpu_util: float = 0.8,
        enable_dynamic_batching: bool = True
    ):
        """
        初始化自适应批处理整理器
        
        Args:
            max_batch_size: 最大批大小
            size_threshold: 尺寸差异阈值（用于分组）
            target_gpu_util: 目标GPU利用率
            enable_dynamic_batching: 是否启用动态批处理
        """
        self.max_batch_size = max_batch_size
        self.size_threshold = size_threshold
        self.target_gpu_util = target_gpu_util
        self.enable_dynamic_batching = enable_dynamic_batching
        
        self.base_collator = BatchCollator(max_group_diff=size_threshold)
        
        logger.info(
            f"AdaptiveBatchCollator initialized: "
            f"max_batch_size={max_batch_size}, "
            f"size_threshold={size_threshold}, "
            f"dynamic_batching={enable_dynamic_batching}"
        )
    
    def group_by_size(
        self,
        images: List[Any],
        max_groups: Optional[int] = None
    ) -> List[List[Any]]:
        """
        按尺寸分组图像
        
        Args:
            images: 图像列表
            max_groups: 最大组数（None表示不限制）
            
        Returns:
            grouped_images: 分组后的图像列表
        """
        if not images:
            return []
        
        # 计算图像尺寸信息
        size_info = []
        for img in images:
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
            elif isinstance(img, torch.Tensor):
                h, w = img.shape[-2:]
            else:
                # PIL Image
                w, h = img.size
            
            aspect_ratio = w / max(h, 1)
            area = w * h
            size_info.append({
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
                'area': area,
                'image': img
            })
        
        # 按宽高比排序
        size_info.sort(key=lambda x: x['aspect_ratio'])
        
        # 分组
        groups = []
        current_group = []
        current_aspect = None
        
        for info in size_info:
            aspect_ratio = info['aspect_ratio']
            
            if current_aspect is None:
                current_aspect = aspect_ratio
                current_group.append(info['image'])
            else:
                # 检查是否应该新建组
                diff = abs(aspect_ratio - current_aspect) / max(current_aspect, 1)
                
                if diff <= self.size_threshold and len(current_group) < self.max_batch_size:
                    current_group.append(info['image'])
                else:
                    # 保存当前组，开始新组
                    groups.append(current_group)
                    current_group = [info['image']]
                    current_aspect = aspect_ratio
                    
                    # 检查组数限制
                    if max_groups and len(groups) >= max_groups:
                        break
        
        # 添加最后一组
        if current_group:
            groups.append(current_group)
        
        logger.debug(
            f"Grouped {len(images)} images into {len(groups)} groups, "
            f"sizes: {[len(g) for g in groups]}"
        )
        
        return groups
    
    def dynamic_padding(
        self,
        images: List[torch.Tensor],
        align_to: int = 32
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        动态填充到最小必要尺寸
        
        Args:
            images: 图像Tensor列表
            align_to: 对齐大小（用于GPU优化）
            
        Returns:
            padded_batch: 填充后的批量Tensor
            padding_info: 填充信息
        """
        if not images:
            raise ValueError("Empty image list")
        
        # 获取批次中的最大尺寸
        max_h = max(img.shape[-2] for img in images)
        max_w = max(img.shape[-1] for img in images)
        
        # 对齐到指定大小（优化GPU计算）
        if align_to > 1:
            max_h = ((max_h + align_to - 1) // align_to) * align_to
            max_w = ((max_w + align_to - 1) // align_to) * align_to
        
        # 创建填充后的批量Tensor
        batch_size = len(images)
        channels = images[0].shape[-3] if images[0].dim() >= 3 else 1
        
        padded_batch = torch.zeros(
            batch_size, channels, max_h, max_w,
            dtype=images[0].dtype,
            device=images[0].device
        )
        
        # 填充每个图像
        original_shapes = []
        for i, img in enumerate(images):
            if img.dim() == 2:
                img = img.unsqueeze(0)  # 添加通道维度
            
            c, h, w = img.shape[-3:]
            padded_batch[i, :, :h, :w] = img
            original_shapes.append((h, w))
        
        padding_info = {
            'padded_size': (max_h, max_w),
            'original_shapes': original_shapes,
            'padding_ratio': (max_h * max_w * batch_size) / sum(h * w for h, w in original_shapes)
        }
        
        logger.debug(
            f"Dynamic padding: {batch_size} images to {max_h}x{max_w}, "
            f"padding_ratio={padding_info['padding_ratio']:.2f}"
        )
        
        return padded_batch, padding_info
    
    def get_optimal_batch_size(
        self,
        image_sizes: List[Tuple[int, int]],
        available_memory: Optional[int] = None
    ) -> int:
        """
        根据图像尺寸和可用内存计算最优批大小
        
        Args:
            image_sizes: 图像尺寸列表 [(H, W), ...]
            available_memory: 可用GPU内存（字节），None表示自动检测
            
        Returns:
            optimal_batch_size: 最优批大小
        """
        if not self.enable_dynamic_batching:
            return self.max_batch_size
        
        if not image_sizes:
            return 1
        
        # 获取可用GPU内存
        if available_memory is None and torch.cuda.is_available():
            available_memory = torch.cuda.mem_get_info()[0]  # 可用内存（字节）
        
        if available_memory is None:
            return self.max_batch_size
        
        # 估算单张图像的内存消耗
        avg_h = sum(h for h, w in image_sizes) / len(image_sizes)
        avg_w = sum(w for h, w in image_sizes) / len(image_sizes)
        
        # 估算每张图像的内存（bytes）
        # 假设: float32 (4 bytes) * 3 channels * H * W * 模型系数（约5倍用于中间激活）
        bytes_per_image = avg_h * avg_w * 3 * 4 * 5
        
        # 计算可容纳的批大小
        target_memory = available_memory * self.target_gpu_util
        optimal_size = int(target_memory / bytes_per_image)
        
        # 限制在合理范围内
        optimal_size = max(1, min(optimal_size, self.max_batch_size))
        
        logger.debug(
            f"Optimal batch size: {optimal_size} "
            f"(avg_size={avg_h:.0f}x{avg_w:.0f}, "
            f"available_mem={available_memory/1024**3:.2f}GB)"
        )
        
        return optimal_size
    
    def collate_adaptive(
        self,
        images: List[Any],
        auto_group: bool = True
    ) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        自适应整理批量数据
        
        Args:
            images: 图像列表
            auto_group: 是否自动分组
            
        Returns:
            batches: [(batch_tensor, batch_info), ...]
        """
        if not images:
            return []
        
        # 如果启用自动分组，先分组
        if auto_group:
            groups = self.group_by_size(images)
        else:
            # 按最大批大小简单分组
            groups = [
                images[i:i + self.max_batch_size]
                for i in range(0, len(images), self.max_batch_size)
            ]
        
        # 处理每个组
        batches = []
        for group_idx, group in enumerate(groups):
            # 转换为Tensor（如果需要）
            tensor_group = []
            for img in group:
                if not isinstance(img, torch.Tensor):
                    # 假设是PIL Image或numpy array
                    if isinstance(img, np.ndarray):
                        tensor = torch.from_numpy(img)
                    else:
                        # PIL Image
                        tensor = torch.from_numpy(np.array(img))
                    
                    # 调整维度顺序：HWC -> CHW
                    if tensor.dim() == 3 and tensor.shape[2] in [1, 3, 4]:
                        tensor = tensor.permute(2, 0, 1)
                    
                    tensor_group.append(tensor)
                else:
                    tensor_group.append(img)
            
            # 动态填充
            batch_tensor, padding_info = self.dynamic_padding(tensor_group)
            
            batch_info = {
                'group_id': group_idx,
                'batch_size': len(group),
                'padding_info': padding_info
            }
            
            batches.append((batch_tensor, batch_info))
        
        logger.info(f"Adaptive collation: {len(images)} images -> {len(batches)} batches")
        
        return batches
