"""
结果解析器
解析模型输出的文本，提取结构化信息
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from .result_data import OCRResult, OCRTextBlock


class ResultParser:
    """
    结果解析器
    负责解析模型输出，支持多种格式
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化结果解析器
        
        Args:
            strict_mode: 严格模式，解析失败时抛出异常而不是降级
        """
        self.strict_mode = strict_mode
    
    def parse(
        self,
        text: str,
        output_format: str = "text",
        confidence_threshold: float = 0.0
    ) -> OCRResult:
        """
        解析文本结果
        
        Args:
            text: 解码后的文本
            output_format: 输出格式 ('json', 'text', 'markdown')
            confidence_threshold: 置信度阈值
            
        Returns:
            OCRResult: 解析后的结构化结果
        """
        if output_format == "json":
            return self._parse_json_format(text, confidence_threshold)
        elif output_format == "markdown":
            return self._parse_markdown_format(text)
        else:
            return self._parse_text_format(text)
    
    def _parse_text_format(self, text: str) -> OCRResult:
        """
        解析纯文本格式
        
        Args:
            text: 输入文本
            
        Returns:
            OCRResult: 只包含文本，无结构化信息
        """
        return OCRResult.from_text(text.strip())
    
    def _parse_json_format(self, text: str, confidence_threshold: float) -> OCRResult:
        """
        解析 JSON 格式
        支持三种 JSON 格式:
        1. 标准格式: {"blocks": [{"text": "...", "bbox": [...], "confidence": ...}]}
        2. 代码块格式: ```json {...} ```
        3. 简化格式: [{"text": "...", ...}]
        
        Args:
            text: 包含 JSON 的文本
            confidence_threshold: 置信度阈值
            
        Returns:
            OCRResult: 解析后的结果
        """
        try:
            # 方法1: 尝试提取代码块中的 JSON
            json_str = self._extract_json_from_code_block(text)
            
            # 方法2: 如果没有代码块，尝试直接解析
            if not json_str:
                json_str = self._extract_json_from_text(text)
            
            if json_str:
                data = json.loads(json_str)
                return self._parse_json_data(data, confidence_threshold)
            else:
                if self.strict_mode:
                    raise ValueError("No valid JSON found in text")
                # 降级到文本格式
                return self._parse_text_format(text)
                
        except (json.JSONDecodeError, ValueError) as e:
            if self.strict_mode:
                raise
            # 降级到文本格式
            return self._parse_text_format(text)
    
    def _extract_json_from_code_block(self, text: str) -> Optional[str]:
        """
        从 Markdown 代码块中提取 JSON
        
        Args:
            text: 输入文本
            
        Returns:
            JSON 字符串或 None
        """
        # 匹配 ```json ... ``` 或 ``` ... ```
        pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """
        从文本中提取 JSON
        
        Args:
            text: 输入文本
            
        Returns:
            JSON 字符串或 None
        """
        # 尝试匹配 [] (数组) 或 {} (对象)
        # 使用贪婪匹配以获取完整的 JSON
        for pattern in [r'\[.*\]', r'\{.*\}']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
        return None
    
    def _parse_json_data(self, data: Any, confidence_threshold: float) -> OCRResult:
        """
        解析 JSON 数据
        
        Args:
            data: 解析的 JSON 对象（可能是 dict 或 list）
            confidence_threshold: 置信度阈值
            
        Returns:
            OCRResult
        """
        text_blocks = []
        
        # 处理简化格式（直接是数组）
        if isinstance(data, list):
            blocks_data = data
        # 处理标准格式
        elif isinstance(data, dict):
            if 'blocks' in data:
                blocks_data = data['blocks']
            else:
                # 尝试其他可能的键名
                blocks_data = data.get('results', data.get('text_blocks', []))
        else:
            blocks_data = []
        
        # 解析每个文本块
        for block_data in blocks_data:
            if not isinstance(block_data, dict):
                continue
                
            # 提取字段
            text = block_data.get('text', '')
            confidence = block_data.get('confidence', 1.0)
            
            # 应用置信度过滤
            if confidence < confidence_threshold:
                continue
            
            # 提取边界框（支持多种字段名）
            bbox = (
                block_data.get('bbox') or
                block_data.get('bounding_box') or
                block_data.get('box') or
                [0, 0, 0, 0]
            )
            
            # 提取语言
            language = block_data.get('language') or block_data.get('lang')
            
            # 创建文本块
            text_block = OCRTextBlock(
                text=text,
                bbox=bbox,
                confidence=confidence,
                language=language,
                metadata=block_data.get('metadata', {})
            )
            text_blocks.append(text_block)
        
        # 创建 OCRResult
        # 如果 data 是字典，提取元数据；如果是列表，使用空元数据
        if isinstance(data, dict):
            return OCRResult(
                text_blocks=text_blocks,
                model_name=data.get('model_name'),
                metadata=data.get('metadata', {})
            )
        else:
            return OCRResult(
                text_blocks=text_blocks
            )
    
    def _parse_markdown_format(self, text: str) -> OCRResult:
        """
        解析 Markdown 格式
        从 Markdown 中提取文本块
        
        Args:
            text: Markdown 文本
            
        Returns:
            OCRResult
        """
        # 提取所有标题和段落
        lines = text.split('\n')
        text_blocks = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 提取标题或段落
            if line.startswith('#'):
                # 标题
                text_content = re.sub(r'^#+\s*', '', line)
            else:
                text_content = line
            
            # 创建文本块（无边界框信息）
            text_block = OCRTextBlock(
                text=text_content,
                bbox=[0, i * 20, 1000, (i + 1) * 20],  # 假设的位置
                confidence=1.0
            )
            text_blocks.append(text_block)
        
        return OCRResult(text_blocks=text_blocks)
    
    def validate_result(self, result: OCRResult) -> bool:
        """
        验证解析结果
        
        Args:
            result: OCR 结果
            
        Returns:
            是否有效
        """
        # 检查是否有文本
        if not result.full_text.strip():
            return False
        
        # 检查文本块的有效性
        for block in result.text_blocks:
            # 检查置信度范围
            if not 0 <= block.confidence <= 1:
                return False
            
            # 检查边界框有效性
            if block.width < 0 or block.height < 0:
                return False
        
        return True
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ResultParser(strict_mode={self.strict_mode})"
