"""
InternVL模型封装
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from .base import VLMModelBase
from typing import Any, Dict, Union, List
from utils.logger import get_logger


logger = get_logger(__name__)


class InternVLModel(VLMModelBase):
    """InternVL模型封装"""
    
    def __init__(self, model_name_or_path: str = "OpenGVLab/InternVL-Chat-V1-5", 
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float16,
                 trust_remote_code: bool = True):
        """
        初始化InternVL模型
        
        Args:
            model_name_or_path: 模型路径或 HuggingFace ID
            device: 运行设备
            torch_dtype: 模型精度
            trust_remote_code: 是否信任远程代码
        """
        super().__init__(model_name_or_path, device, torch_dtype, trust_remote_code)
        self.load_model()
        self.load_tokenizer()
    
    def load_model(self) -> None:
        """加载InternVL模型"""
        try:
            logger.info(f"Loading InternVL model: {self.model_name_or_path}")
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != 'cpu' else None
            )
            
            if self.device == 'cpu':
                self.model = self.model.to('cpu')
                
            logger.info("✓ InternVL model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def load_processor(self) -> None:
        """加载 processor（InternVL 使用 tokenizer）"""
        # InternVL 使用 tokenizer 而不是独立的 processor
        self.processor = None
        logger.info("InternVL uses tokenizer directly")
    
    def load_tokenizer(self) -> None:
        """加载InternVL tokenizer"""
        try:
            logger.info(f"Loading InternVL tokenizer: {self.model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code
            )
            logger.info("✓ InternVL tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load tokenizer: {e}")
            raise RuntimeError(f"Tokenizer loading failed: {e}")
    
    def prepare_inputs(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备模型输入
        
        Args:
            image: PIL Image 或图像路径
            prompt: 文本提示
            **kwargs: 其他参数
            
        Returns:
            Dict: 模型输入字典
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # InternVL 特定的输入准备逻辑
        # TODO: 实现具体的输入准备
        return {
            'images': [image],
            'prompt': prompt
        }
    
    def decode(self, output_ids: torch.Tensor, **kwargs) -> str:
        """
        解码输出
        
        Args:
            output_ids: 模型输出的 token ids
            **kwargs: 解码参数
            
        Returns:
            str: 解码后的文本
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=kwargs.get('skip_special_tokens', True)
        )
        return text.strip()
    
    def generate(self, inputs: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        生成输出
        
        Args:
            inputs: 模型输入字典
            **kwargs: 生成参数
            
        Returns:
            生成的token ids
        """
        logger.info("Generating output with InternVL...")
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 512),
            )
            
            logger.info("Generation completed")
            return outputs
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
