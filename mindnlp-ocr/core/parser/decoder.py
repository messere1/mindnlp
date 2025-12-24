"""
Token 解码器
将模型输出的 token ID 转换为文本
"""

from typing import List, Union, Any
import re


class TokenDecoder:
    """
    Token 解码器
    负责将模型输出的 token IDs 解码为文本
    """
    
    def __init__(
        self,
        tokenizer: Any,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ):
        """
        初始化解码器
        
        Args:
            tokenizer: 分词器实例（transformers tokenizer）
            skip_special_tokens: 是否跳过特殊 token
            clean_up_tokenization_spaces: 是否清理分词空格
        """
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
    
    def decode(
        self,
        token_ids: Union[List[int], Any],
        **kwargs
    ) -> str:
        """
        解码单个序列
        
        Args:
            token_ids: Token ID 列表或 tensor
            **kwargs: 额外的解码参数
            
        Returns:
            解码后的文本
        """
        # 合并默认参数
        decode_kwargs = {
            'skip_special_tokens': self.skip_special_tokens,
            'clean_up_tokenization_spaces': self.clean_up_tokenization_spaces,
            **kwargs
        }
        
        # 如果是 tensor，转换为列表
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # 解码
        text = self.tokenizer.decode(token_ids, **decode_kwargs)
        
        # 后处理
        return self._post_process(text)
    
    def batch_decode(
        self,
        token_ids_list: Union[List[List[int]], Any],
        **kwargs
    ) -> List[str]:
        """
        批量解码
        
        Args:
            token_ids_list: Token ID 列表的列表或 tensor
            **kwargs: 额外的解码参数
            
        Returns:
            解码后的文本列表
        """
        # 合并默认参数
        decode_kwargs = {
            'skip_special_tokens': self.skip_special_tokens,
            'clean_up_tokenization_spaces': self.clean_up_tokenization_spaces,
            **kwargs
        }
        
        # 如果是 tensor，转换为列表
        if hasattr(token_ids_list, 'tolist'):
            token_ids_list = token_ids_list.tolist()
        
        # 批量解码
        texts = self.tokenizer.batch_decode(token_ids_list, **decode_kwargs)
        
        # 后处理
        return [self._post_process(text) for text in texts]
    
    def _post_process(self, text: str) -> str:
        """
        后处理解码的文本
        
        Args:
            text: 原始解码文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text
    
    def decode_with_metadata(
        self,
        token_ids: Union[List[int], Any],
        return_offsets: bool = False,
        **kwargs
    ) -> dict:
        """
        解码并返回元数据
        
        Args:
            token_ids: Token ID 列表或 tensor
            return_offsets: 是否返回字符偏移量
            **kwargs: 额外的解码参数
            
        Returns:
            包含文本和元数据的字典
        """
        # 解码文本
        text = self.decode(token_ids, **kwargs)
        
        result = {
            'text': text,
            'num_tokens': len(token_ids) if isinstance(token_ids, list) else len(token_ids.tolist()),
            'num_characters': len(text)
        }
        
        # 如果需要偏移量
        if return_offsets:
            result['offsets'] = self._compute_offsets(token_ids, text)
        
        return result
    
    def _compute_offsets(self, token_ids: List[int], text: str) -> List[tuple]:
        """
        计算字符偏移量
        
        Args:
            token_ids: Token ID 列表
            text: 解码后的文本
            
        Returns:
            偏移量列表 [(start, end), ...]
        """
        # 简化实现：返回空列表
        # 实际实现需要根据 tokenizer 的特性来计算精确的偏移量
        return []
    
    def validate_output(self, text: str) -> bool:
        """
        验证解码输出
        
        Args:
            text: 解码的文本
            
        Returns:
            是否有效
        """
        # 检查文本是否为空
        if not text or not text.strip():
            return False
        
        # 检查是否包含过多的特殊字符
        special_char_ratio = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text)) / max(len(text), 1)
        if special_char_ratio > 0.5:  # 特殊字符超过 50%
            return False
        
        return True
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"TokenDecoder("
            f"skip_special_tokens={self.skip_special_tokens}, "
            f"clean_up_tokenization_spaces={self.clean_up_tokenization_spaces})"
        )
