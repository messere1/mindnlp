"""
OCR 结果数据类
定义 OCR 结果的数据结构
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from PIL import Image


@dataclass
class OCRTextBlock:
    """
    单个 OCR 文本块
    
    Attributes:
        text: 识别的文本内容
        bbox: 边界框 [x1, y1, x2, y2]，左上角和右下角坐标
        confidence: 置信度分数 (0-1)
        language: 语言标识（可选）
        metadata: 额外的元数据
    """
    text: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 1.0
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据"""
        if not isinstance(self.bbox, list) or len(self.bbox) != 4:
            raise ValueError(f"bbox must be a list of 4 numbers, got {self.bbox}")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
    
    @property
    def x1(self) -> float:
        """左上角 x 坐标"""
        return self.bbox[0]
    
    @property
    def y1(self) -> float:
        """左上角 y 坐标"""
        return self.bbox[1]
    
    @property
    def x2(self) -> float:
        """右下角 x 坐标"""
        return self.bbox[2]
    
    @property
    def y2(self) -> float:
        """右下角 y 坐标"""
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        """边界框宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """边界框高度"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """边界框面积"""
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        """中心点坐标 (x, y)"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'text': self.text,
            'bbox': self.bbox,
            'confidence': self.confidence
        }
        if self.language:
            result['language'] = self.language
        if self.metadata:
            result['metadata'] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRTextBlock':
        """从字典创建"""
        return cls(
            text=data['text'],
            bbox=data['bbox'],
            confidence=data.get('confidence', 1.0),
            language=data.get('language'),
            metadata=data.get('metadata', {})
        )


@dataclass
class OCRResult:
    """
    完整的 OCR 结果
    
    Attributes:
        text_blocks: 识别的文本块列表
        raw_text: 原始文本（未结构化）
        image_size: 原始图像尺寸 (width, height)
        processing_time: 处理耗时（秒）
        model_name: 使用的模型名称
        metadata: 额外的元数据
    """
    text_blocks: List[OCRTextBlock] = field(default_factory=list)
    raw_text: str = ""
    image_size: Optional[tuple] = None
    processing_time: Optional[float] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """返回文本块数量"""
        return len(self.text_blocks)
    
    def __iter__(self):
        """迭代文本块"""
        return iter(self.text_blocks)
    
    def __getitem__(self, index: int) -> OCRTextBlock:
        """通过索引访问文本块"""
        return self.text_blocks[index]
    
    @property
    def full_text(self) -> str:
        """拼接所有文本块的文本"""
        if self.raw_text:
            return self.raw_text
        return '\n'.join(block.text for block in self.text_blocks)
    
    @property
    def average_confidence(self) -> float:
        """平均置信度"""
        if not self.text_blocks:
            return 0.0
        return sum(block.confidence for block in self.text_blocks) / len(self.text_blocks)
    
    @property
    def total_characters(self) -> int:
        """总字符数"""
        return sum(len(block.text) for block in self.text_blocks)
    
    def filter_by_confidence(self, min_confidence: float) -> 'OCRResult':
        """
        按置信度过滤文本块
        
        Args:
            min_confidence: 最小置信度阈值
            
        Returns:
            过滤后的新 OCRResult
        """
        filtered_blocks = [
            block for block in self.text_blocks 
            if block.confidence >= min_confidence
        ]
        
        return OCRResult(
            text_blocks=filtered_blocks,
            raw_text=self.raw_text,
            image_size=self.image_size,
            processing_time=self.processing_time,
            model_name=self.model_name,
            metadata=self.metadata.copy()
        )
    
    def sort_blocks(self, key='position', reverse=False) -> 'OCRResult':
        """
        排序文本块
        
        Args:
            key: 排序键 ('position', 'confidence', 'length')
            reverse: 是否倒序
            
        Returns:
            排序后的新 OCRResult
        """
        if key == 'position':
            # 按位置排序：先按 y 坐标，再按 x 坐标
            sorted_blocks = sorted(
                self.text_blocks,
                key=lambda b: (b.y1, b.x1),
                reverse=reverse
            )
        elif key == 'confidence':
            sorted_blocks = sorted(
                self.text_blocks,
                key=lambda b: b.confidence,
                reverse=reverse
            )
        elif key == 'length':
            sorted_blocks = sorted(
                self.text_blocks,
                key=lambda b: len(b.text),
                reverse=reverse
            )
        else:
            raise ValueError(f"Unknown sort key: {key}")
        
        return OCRResult(
            text_blocks=sorted_blocks,
            raw_text=self.raw_text,
            image_size=self.image_size,
            processing_time=self.processing_time,
            model_name=self.model_name,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'text_blocks': [block.to_dict() for block in self.text_blocks],
            'raw_text': self.raw_text,
            'full_text': self.full_text,
            'num_blocks': len(self.text_blocks),
            'total_characters': self.total_characters,
            'average_confidence': self.average_confidence
        }
        
        if self.image_size:
            result['image_size'] = self.image_size
        if self.processing_time:
            result['processing_time'] = self.processing_time
        if self.model_name:
            result['model_name'] = self.model_name
        if self.metadata:
            result['metadata'] = self.metadata
            
        return result
    
    def to_json(self) -> Dict[str, Any]:
        """转换为 JSON 格式（兼容 API 输出）"""
        return {
            'results': [block.to_dict() for block in self.text_blocks],
            'summary': {
                'total_blocks': len(self.text_blocks),
                'total_characters': self.total_characters,
                'average_confidence': self.average_confidence
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResult':
        """从字典创建"""
        text_blocks = [
            OCRTextBlock.from_dict(block_data)
            for block_data in data.get('text_blocks', [])
        ]
        
        return cls(
            text_blocks=text_blocks,
            raw_text=data.get('raw_text', ''),
            image_size=tuple(data['image_size']) if 'image_size' in data else None,
            processing_time=data.get('processing_time'),
            model_name=data.get('model_name'),
            metadata=data.get('metadata', {})
        )
    
    @classmethod
    def from_text(cls, text: str) -> 'OCRResult':
        """从纯文本创建（无结构化信息）"""
        return cls(raw_text=text)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"OCRResult(blocks={len(self.text_blocks)}, "
            f"chars={self.total_characters}, "
            f"avg_conf={self.average_confidence:.2f})"
        )
