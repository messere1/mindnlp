"""
输出格式化器
应用坐标映射、置信度过滤、去重等后处理
"""

from typing import Dict, Any, List, Optional, Tuple
from .result_data import OCRResult, OCRTextBlock


class OutputFormatter:
    """
    输出格式化器
    负责：
    1. 坐标映射（还原到原始图像坐标）
    2. 置信度过滤
    3. 结果排序
    4. 去重
    """
    
    def __init__(
        self,
        min_confidence: float = 0.0,
        enable_deduplication: bool = True,
        iou_threshold: float = 0.5
    ):
        """
        初始化格式化器
        
        Args:
            min_confidence: 最小置信度阈值
            enable_deduplication: 是否启用去重
            iou_threshold: IoU 阈值用于去重
        """
        self.min_confidence = min_confidence
        self.enable_deduplication = enable_deduplication
        self.iou_threshold = iou_threshold
    
    def format(
        self,
        result: OCRResult,
        transform_info: Optional[Dict[str, Any]] = None,
        sort_by: str = 'position'
    ) -> OCRResult:
        """
        格式化 OCR 结果
        
        Args:
            result: 原始 OCR 结果
            transform_info: 变换信息（padding, scale）
            sort_by: 排序方式 ('position', 'confidence', 'length')
            
        Returns:
            格式化后的 OCR 结果
        """
        # 1. 应用坐标映射
        if transform_info:
            result = self.apply_coordinate_mapping(result, transform_info)
        
        # 2. 置信度过滤
        if self.min_confidence > 0:
            result = result.filter_by_confidence(self.min_confidence)
        
        # 3. 去重
        if self.enable_deduplication:
            result = self.deduplicate(result)
        
        # 4. 排序
        result = result.sort_blocks(key=sort_by)
        
        return result
    
    def apply_coordinate_mapping(
        self,
        result: OCRResult,
        transform_info: Dict[str, Any]
    ) -> OCRResult:
        """
        应用坐标映射，还原到原始图像坐标
        
        Args:
            result: OCR 结果
            transform_info: 变换信息
                - padding: (left, top, right, bottom)
                - scale: (scale_x, scale_y)
                - original_size: (width, height)
                
        Returns:
            坐标映射后的结果
        """
        padding = transform_info.get('padding', (0, 0, 0, 0))
        scale = transform_info.get('scale', (1.0, 1.0))
        
        # 提取 padding 和 scale
        pad_left, pad_top = padding[0], padding[1]
        scale_x, scale_y = scale
        
        # 映射每个文本块的坐标
        mapped_blocks = []
        for block in result.text_blocks:
            # 还原 padding
            x1 = block.x1 - pad_left
            y1 = block.y1 - pad_top
            x2 = block.x2 - pad_left
            y2 = block.y2 - pad_top
            
            # 还原 scale
            x1 = x1 / scale_x
            y1 = y1 / scale_y
            x2 = x2 / scale_x
            y2 = y2 / scale_y
            
            # 创建新的文本块
            mapped_block = OCRTextBlock(
                text=block.text,
                bbox=[x1, y1, x2, y2],
                confidence=block.confidence,
                language=block.language,
                metadata=block.metadata.copy()
            )
            mapped_blocks.append(mapped_block)
        
        # 创建新的结果
        return OCRResult(
            text_blocks=mapped_blocks,
            raw_text=result.raw_text,
            image_size=transform_info.get('original_size'),
            processing_time=result.processing_time,
            model_name=result.model_name,
            metadata=result.metadata.copy()
        )
    
    def deduplicate(self, result: OCRResult) -> OCRResult:
        """
        去重：移除重叠的文本块
        
        Args:
            result: OCR 结果
            
        Returns:
            去重后的结果
        """
        if len(result.text_blocks) <= 1:
            return result
        
        # 按置信度排序（高置信度优先保留）
        sorted_blocks = sorted(
            result.text_blocks,
            key=lambda b: b.confidence,
            reverse=True
        )
        
        # 保留的文本块
        kept_blocks = []
        
        for block in sorted_blocks:
            # 检查是否与已保留的块重叠
            is_duplicate = False
            for kept_block in kept_blocks:
                iou = self._compute_iou(block, kept_block)
                if iou > self.iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_blocks.append(block)
        
        # 创建新结果
        return OCRResult(
            text_blocks=kept_blocks,
            raw_text=result.raw_text,
            image_size=result.image_size,
            processing_time=result.processing_time,
            model_name=result.model_name,
            metadata=result.metadata.copy()
        )
    
    def _compute_iou(self, block1: OCRTextBlock, block2: OCRTextBlock) -> float:
        """
        计算两个文本块的 IoU (Intersection over Union)
        
        Args:
            block1: 第一个文本块
            block2: 第二个文本块
            
        Returns:
            IoU 值 (0-1)
        """
        # 计算交集
        x1 = max(block1.x1, block2.x1)
        y1 = max(block1.y1, block2.y1)
        x2 = min(block1.x2, block2.x2)
        y2 = min(block1.y2, block2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0  # 没有交集
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集
        area1 = block1.area
        area2 = block2.area
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def filter_small_blocks(
        self,
        result: OCRResult,
        min_width: float = 5,
        min_height: float = 5
    ) -> OCRResult:
        """
        过滤过小的文本块
        
        Args:
            result: OCR 结果
            min_width: 最小宽度
            min_height: 最小高度
            
        Returns:
            过滤后的结果
        """
        filtered_blocks = [
            block for block in result.text_blocks
            if block.width >= min_width and block.height >= min_height
        ]
        
        return OCRResult(
            text_blocks=filtered_blocks,
            raw_text=result.raw_text,
            image_size=result.image_size,
            processing_time=result.processing_time,
            model_name=result.model_name,
            metadata=result.metadata.copy()
        )
    
    def merge_adjacent_blocks(
        self,
        result: OCRResult,
        max_distance: float = 10
    ) -> OCRResult:
        """
        合并相邻的文本块
        
        Args:
            result: OCR 结果
            max_distance: 最大距离阈值
            
        Returns:
            合并后的结果
        """
        if len(result.text_blocks) <= 1:
            return result
        
        # 按位置排序
        sorted_blocks = sorted(result.text_blocks, key=lambda b: (b.y1, b.x1))
        
        # 合并
        merged_blocks = [sorted_blocks[0]]
        
        for block in sorted_blocks[1:]:
            last_block = merged_blocks[-1]
            
            # 检查是否在同一行且距离足够近
            if (abs(block.y1 - last_block.y1) < max_distance and
                block.x1 - last_block.x2 < max_distance):
                # 合并文本
                merged_text = last_block.text + ' ' + block.text
                merged_bbox = [
                    last_block.x1,
                    min(last_block.y1, block.y1),
                    block.x2,
                    max(last_block.y2, block.y2)
                ]
                merged_confidence = (last_block.confidence + block.confidence) / 2
                
                # 更新最后一个块
                merged_blocks[-1] = OCRTextBlock(
                    text=merged_text,
                    bbox=merged_bbox,
                    confidence=merged_confidence
                )
            else:
                # 添加新块
                merged_blocks.append(block)
        
        return OCRResult(
            text_blocks=merged_blocks,
            raw_text=result.raw_text,
            image_size=result.image_size,
            processing_time=result.processing_time,
            model_name=result.model_name,
            metadata=result.metadata.copy()
        )
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"OutputFormatter("
            f"min_confidence={self.min_confidence}, "
            f"dedup={self.enable_deduplication}, "
            f"iou_threshold={self.iou_threshold})"
        )
