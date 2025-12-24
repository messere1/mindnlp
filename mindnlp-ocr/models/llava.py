"""
LLaVA模型封装
支持LLaVA视觉指令遵循模型
"""

import torch
from typing import Dict, Any, Optional
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from .base import VLMModelBase
from utils.logger import get_logger


logger = get_logger(__name__)


class LLaVAModel(VLMModelBase):
    """
    LLaVA模型封装
    
    特性:
    - 视觉指令遵循能力强
    - 推理速度快
    - 支持多种分辨率
    """
    
    # 默认模型名称
    DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "llava-hf/llava-1.5-7b-hf",
        "llava-hf/llava-1.5-13b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-vicuna-7b-hf",
    ]
    
    def __init__(
        self, 
        model_name_or_path: Optional[str] = None, 
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        low_cpu_mem_usage: bool = True,
        **kwargs
    ):
        """
        初始化LLaVA模型
        
        Args:
            model_name_or_path: 模型名称或路径
            device: 运行设备 (cuda/cpu)
            torch_dtype: 数据类型
            low_cpu_mem_usage: 是否使用低CPU内存模式
        """
        model_name_or_path = model_name_or_path or self.DEFAULT_MODEL
        super().__init__(model_name_or_path, device, torch_dtype=torch_dtype, **kwargs)
        
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.processor = None
        
        logger.info(f"LLaVA model initialized: {model_name_or_path}")
    
    def load_model(self):
        """加载LLaVA模型"""
        try:
            logger.info(f"Loading LLaVA model from {self.model_name_or_path}...")
            
            # 加载模型
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
                device_map=self.device if self.device != "cpu" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("LLaVA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA model: {e}")
            raise
    
    def load_tokenizer(self):
        """加载LLaVA处理器（包含tokenizer和图像处理器）"""
        try:
            logger.info(f"Loading LLaVA processor from {self.model_name_or_path}...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
            self.tokenizer = self.processor.tokenizer
            
            logger.info("LLaVA processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA processor: {e}")
            raise
    
    def load_processor(self):
        """加载处理器（与load_tokenizer相同）"""
        self.load_tokenizer()
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        解码token IDs为文本
        
        Args:
            token_ids: Token IDs
            skip_special_tokens: 是否跳过特殊token
            
        Returns:
            str: 解码后的文本
        """
        if self.processor is None:
            self.load_tokenizer()
        
        return self.processor.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def prepare_inputs(
        self, 
        image: Image.Image, 
        prompt: str
    ) -> Dict[str, Any]:
        """
        准备模型输入
        
        Args:
            image: PIL图像
            prompt: 文本提示
            
        Returns:
            Dict: 模型输入字典
        """
        return self.preprocess(image, prompt)
    
    def preprocess(
        self, 
        image: Image.Image, 
        prompt: str,
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """
        预处理图像和文本
        
        Args:
            image: PIL图像
            prompt: 文本提示
            return_tensors: 返回tensor类型
            
        Returns:
            Dict: 包含模型输入的字典
        """
        if self.processor is None:
            self.load_tokenizer()
        
        # LLaVA需要特殊的prompt格式
        # 格式: USER: <image>\n{prompt}\nASSISTANT:
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        try:
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors=return_tensors,
                padding=True
            )
            
            # 将输入移到正确的设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def generate(
        self, 
        inputs: Dict[str, Any],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.2,
        top_p: float = 1.0,
        num_beams: int = 1,
        **kwargs
    ) -> str:
        """
        生成文本
        
        Args:
            inputs: 模型输入
            max_new_tokens: 最大生成token数
            do_sample: 是否采样
            temperature: 采样温度
            top_p: nucleus采样参数
            num_beams: beam search数量
            **kwargs: 其他生成参数
            
        Returns:
            str: 生成的文本
        """
        if self.model is None:
            self.load_model()
        
        if self.processor is None:
            self.load_tokenizer()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    **kwargs
                )
            
            # 解码输出
            generated_text = self.processor.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # 提取ASSISTANT的回复
            # LLaVA的输出格式: USER: ... ASSISTANT: {response}
            if "ASSISTANT:" in generated_text:
                generated_text = generated_text.split("ASSISTANT:")[-1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def predict(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        端到端预测
        
        Args:
            image: PIL图像
            prompt: 文本提示
            max_new_tokens: 最大生成token数
            **kwargs: 其他生成参数
            
        Returns:
            str: 预测结果
        """
        # 预处理
        inputs = self.preprocess(image, prompt)
        
        # 生成
        result = self.generate(inputs, max_new_tokens=max_new_tokens, **kwargs)
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        return {
            "model_name": self.model_name_or_path,
            "model_type": "LLaVA",
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "supported_models": self.SUPPORTED_MODELS,
            "default_resolution": "336x336",
            "capabilities": [
                "视觉问答",
                "图像描述",
                "指令遵循",
                "OCR识别"
            ]
        }
    
    def __repr__(self) -> str:
        return f"LLaVAModel(model_name={self.model_name_or_path}, device={self.device})"
