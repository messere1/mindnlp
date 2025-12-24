"""
Qwen2-VL模型封装
使用 transformers 库实现
"""

import torch
from typing import Dict, Union, List, Any
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

from .base import VLMModelBase
from utils.logger import get_logger


logger = get_logger(__name__)


class Qwen2VLModel(VLMModelBase):
    """
    Qwen2-VL模型封装
    
    基于 transformers 库的 Qwen2-VL-7B-Instruct 模型
    支持图像+文本输入的视觉语言模型
    """
    
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True
    ):
        """
        初始化 Qwen2-VL 模型
        
        Args:
            model_name_or_path: 模型路径或 HuggingFace ID
            device: 运行设备 ('cuda', 'cpu', 'auto')
            torch_dtype: 模型精度（默认 float16 以节省显存）
            trust_remote_code: 是否信任远程代码（Qwen2-VL 需要）
        """
        super().__init__(model_name_or_path, device, torch_dtype, trust_remote_code)
        
        # 加载模型组件
        self.load_model()
        self.load_processor()
        self.load_tokenizer()
        
        # 设置为评估模式
        self.eval()
        
        logger.info("Qwen2VLModel initialization completed")
    
    def load_model(self) -> None:
        """
        加载 Qwen2-VL 模型
        
        使用 Qwen2VLForConditionalGeneration
        默认使用 float16 以降低显存占用（约14GB）
        """
        try:
            logger.info(f"Loading Qwen2-VL model: {self.model_name_or_path}")
            logger.info(f"Using dtype: {self.torch_dtype}, device_map: {self.device}")
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != 'cpu' else None,
                trust_remote_code=self.trust_remote_code
            )
            
            # 如果是 CPU，需要手动移动
            if self.device == 'cpu':
                self.model = self.model.to('cpu')
            
            logger.info("✓ Qwen2-VL model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load Qwen2-VL model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def load_processor(self) -> None:
        """
        加载 Qwen2-VL processor
        
        Processor 用于处理图像+文本输入，转换为模型所需格式
        """
        try:
            logger.info(f"Loading Qwen2-VL processor: {self.model_name_or_path}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info("✓ Qwen2-VL processor loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to load processor: {e}")
            raise RuntimeError(f"Processor loading failed: {e}")
    
    def load_tokenizer(self) -> None:
        """
        加载 Qwen2-VL tokenizer
        
        Tokenizer 用于文本编码和解码
        """
        try:
            logger.info(f"Loading Qwen2-VL tokenizer: {self.model_name_or_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info("✓ Qwen2-VL tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to load tokenizer: {e}")
            raise RuntimeError(f"Tokenizer loading failed: {e}")
    
    def prepare_inputs(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        准备 Qwen2-VL 模型输入
        
        使用 processor.apply_chat_template() 和 messages 格式
        
        Args:
            image: PIL Image 对象或图像路径
            prompt: 文本提示词
            **kwargs: 其他参数
            
        Returns:
            Dict[str, torch.Tensor]: 模型输入字典
            
        Example:
            >>> model = Qwen2VLModel()
            >>> image = Image.open("test.jpg")
            >>> prompt = "识别图像中的文字"
            >>> inputs = model.prepare_inputs(image, prompt)
        """
        try:
            # 如果是路径，加载图像
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            # 构建 messages 格式（Qwen2-VL 要求）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 应用 chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 处理图像和文本
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )
            
            # 移动到模型设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            logger.debug(f"Prepared inputs with keys: {list(inputs.keys())}")
            return inputs
            
        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            raise RuntimeError(f"Input preparation failed: {e}")
    
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
        生成输出 token IDs
        
        Args:
            inputs: 模型输入字典（来自 prepare_inputs）
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（控制随机性）
            top_p: nucleus sampling 参数
            do_sample: 是否使用采样（False 为贪婪解码）
            **kwargs: 其他生成参数
            
        Returns:
            torch.Tensor: 生成的 token IDs [batch_size, sequence_length]
        """
        try:
            logger.info("Generating with Qwen2-VL...")
            logger.debug(f"Generation params: max_new_tokens={max_new_tokens}, "
                        f"temperature={temperature}, top_p={top_p}, do_sample={do_sample}")
            
            # 设置生成配置
            generation_config = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': do_sample,
                **kwargs
            }
            
            # 使用 torch.no_grad() 节省显存
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            logger.info(f"✓ Generation completed, output shape: {outputs.shape}")
            return outputs
            
        except Exception as e:
            logger.error(f"✗ Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> str:
        """
        解码 token IDs 为文本
        
        Args:
            token_ids: 生成的 token IDs [batch_size, sequence_length]
            skip_special_tokens: 是否跳过特殊 token（如 <eos>, <pad>）
            
        Returns:
            str: 解码后的文本
        """
        try:
            # 如果是批次，只取第一个
            if len(token_ids.shape) > 1:
                token_ids = token_ids[0]
            
            # 解码
            decoded_text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens
            )
            
            logger.debug(f"Decoded text length: {len(decoded_text)} characters")
            return decoded_text
            
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise RuntimeError(f"Decoding failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取 Qwen2-VL 模型信息
        
        Returns:
            Dict[str, Any]: 模型详细信息
        """
        info = super().get_model_info()
        
        # 添加 Qwen2-VL 特定信息
        if self.model is not None:
            try:
                config = self.model.config
                info.update({
                    'model_type': 'Qwen2-VL',
                    'hidden_size': getattr(config, 'hidden_size', None),
                    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
                    'num_attention_heads': getattr(config, 'num_attention_heads', None),
                    'vocab_size': getattr(config, 'vocab_size', None),
                })
            except Exception as e:
                logger.warning(f"Failed to get config details: {e}")
        
        return info
    
    def __repr__(self) -> str:
        return (
            f"Qwen2VLModel("
            f"model='{self.model_name_or_path}', "
            f"device='{self.device}', "
            f"dtype={self.torch_dtype})"
        )
