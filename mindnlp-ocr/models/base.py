"""
VLM模型基类
定义统一的模型接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
from PIL import Image
from utils.logger import get_logger


logger = get_logger(__name__)


class VLMModelBase(ABC):
    """VLM模型抽象基类"""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True
    ):
        """
        初始化VLM模型
        
        Args:
            model_name_or_path: 模型名称或路径（HuggingFace ID 或本地路径）
            device: 运行设备 ('cuda', 'cpu', 'auto')
            torch_dtype: 模型精度（默认 float16）
            trust_remote_code: 是否信任远程代码
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        
        logger.info(f"Initializing {self.__class__.__name__} with model: {model_name_or_path}")
        logger.info(f"Device: {device}, dtype: {torch_dtype}")
    
    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型
        
        Raises:
            RuntimeError: 模型加载失败
        """
        pass
    
    @abstractmethod
    def load_tokenizer(self) -> None:
        """
        加载tokenizer
        
        Raises:
            RuntimeError: Tokenizer加载失败
        """
        pass
    
    @abstractmethod
    def load_processor(self) -> None:
        """
        加载processor（用于处理图像+文本输入）
        
        Raises:
            RuntimeError: Processor加载失败
        """
        pass
    
    @abstractmethod
    def prepare_inputs(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        准备模型输入
        
        Args:
            image: PIL图像或图像路径
            prompt: 文本提示
            **kwargs: 其他参数
            
        Returns:
            Dict[str, torch.Tensor]: 模型输入字典
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        生成输出token IDs
        
        Args:
            inputs: 模型输入
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            do_sample: 是否使用采样
            **kwargs: 其他生成参数
            
        Returns:
            torch.Tensor: 生成的token IDs
        """
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> str:
        """
        解码token IDs为文本
        
        Args:
            token_ids: token IDs
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            str: 解码后的文本
        """
        pass
    
    def ocr(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        完整的OCR流程：准备输入 -> 生成 -> 解码
        
        Args:
            image: PIL图像或图像路径
            prompt: 文本提示
            max_new_tokens: 最大生成token数
            **kwargs: 其他参数
            
        Returns:
            str: OCR识别结果
        """
        logger.info("Starting OCR process...")
        
        # 1. 准备输入
        inputs = self.prepare_inputs(image, prompt, **kwargs)
        
        # 2. 生成token IDs
        output_ids = self.generate(inputs, max_new_tokens=max_new_tokens, **kwargs)
        
        # 3. 解码
        result = self.decode(output_ids)
        
        logger.info("OCR process completed")
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        info = {
            'model_name': self.model_name_or_path,
            'device': self.device,
            'dtype': str(self.torch_dtype),
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'processor_loaded': self.processor is not None,
        }
        
        # 如果模型已加载，添加更多信息
        if self.model is not None:
            try:
                info['model_class'] = self.model.__class__.__name__
                info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
                info['trainable_parameters'] = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
            except Exception as e:
                logger.warning(f"Failed to get model details: {e}")
        
        return info
    
    def to(self, device: str) -> None:
        """
        移动模型到指定设备
        
        Args:
            device: 目标设备
        """
        if self.model is not None:
            logger.info(f"Moving model to {device}")
            self.model = self.model.to(device)
            self.device = device
            logger.info(f"Model moved to {device}")
        else:
            logger.warning("Model not loaded, cannot move to device")
    
    def eval(self) -> None:
        """设置模型为评估模式"""
        if self.model is not None:
            self.model.eval()
            logger.info("Model set to evaluation mode")
        else:
            logger.warning("Model not loaded, cannot set to eval mode")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name_or_path}', "
            f"device='{self.device}', "
            f"dtype={self.torch_dtype})"
        )
