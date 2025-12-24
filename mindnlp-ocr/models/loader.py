"""
模型加载器
统一的模型加载入口，支持多种VLM模型
支持多模型管理和动态切换
"""

import os
import torch
from typing import Dict, Optional, List, Union
from pathlib import Path

from .base import VLMModelBase
from .qwen2vl import Qwen2VLModel
from .internvl import InternVLModel
from .llava import LLaVAModel
from utils.logger import get_logger


logger = get_logger(__name__)


class ModelLoader:
    """
    模型加载器
    
    提供统一的模型加载接口，支持：
    - HuggingFace模型ID（如 "Qwen/Qwen2-VL-7B-Instruct"）
    - 本地模型路径
    - 多种VLM模型类型
    """
    
    # 支持的模型映射
    MODEL_MAPPING = {
        'qwen2-vl': Qwen2VLModel,
        'qwen2vl': Qwen2VLModel,
        'qwen': Qwen2VLModel,
        'internvl': InternVLModel,
        'llava': LLaVAModel,
        'llava-1.5': LLaVAModel,
        'llava-1.6': LLaVAModel,
    }
    
    # 默认模型
    DEFAULT_MODEL = 'Qwen/Qwen2-VL-7B-Instruct'
    
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_type: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True
    ):
        """
        初始化模型加载器
        
        Args:
            model_name_or_path: 模型名称或路径
                - HuggingFace ID: "Qwen/Qwen2-VL-7B-Instruct"
                - 本地路径: "/path/to/model"
                - None: 使用默认模型
            model_type: 模型类型（'qwen2-vl', 'internvl'等）
                - None: 自动检测
            device: 运行设备 ('cuda', 'cpu', 'auto')
            torch_dtype: 模型精度（默认 float16）
            trust_remote_code: 是否信任远程代码
        """
        self.model_name_or_path = model_name_or_path or self.DEFAULT_MODEL
        self.model_type = model_type
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        
        self.model_instance: Optional[VLMModelBase] = None
        
        logger.info(f"ModelLoader initialized")
        logger.info(f"  Model: {self.model_name_or_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {self.torch_dtype}")
    
    def _setup_device(self, device: str) -> str:
        """
        设置运行设备
        
        Args:
            device: 设备字符串 ('cuda', 'cpu', 'auto')
            
        Returns:
            str: 实际使用的设备
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Auto-detected CUDA, using GPU")
            else:
                device = 'cpu'
                logger.warning("CUDA not available, using CPU")
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def _detect_model_type(self, model_name_or_path: str) -> str:
        """
        自动检测模型类型
        
        Args:
            model_name_or_path: 模型名称或路径
            
        Returns:
            str: 检测到的模型类型
        """
        name_lower = model_name_or_path.lower()
        
        # 检查是否包含关键词
        if 'qwen2-vl' in name_lower or 'qwen2vl' in name_lower:
            return 'qwen2-vl'
        elif 'qwen' in name_lower:
            return 'qwen2-vl'  # 默认使用 Qwen2-VL
        elif 'internvl' in name_lower:
            return 'internvl'
        elif 'llava' in name_lower:
            return 'llava'
        else:
            logger.warning(
                f"Cannot detect model type from '{model_name_or_path}', "
                f"using 'qwen2-vl' as default"
            )
            return 'qwen2-vl'
    
    def _get_model_class(self, model_type: str) -> type:
        """
        获取模型类
        
        Args:
            model_type: 模型类型
            
        Returns:
            type: 模型类
            
        Raises:
            ValueError: 不支持的模型类型
        """
        model_class = self.MODEL_MAPPING.get(model_type)
        
        if model_class is None:
            supported = ', '.join(self.MODEL_MAPPING.keys())
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: {supported}"
            )
        
        return model_class
    
    def _validate_model_path(self, path: str) -> bool:
        """
        验证本地模型路径是否有效
        
        Args:
            path: 模型路径
            
        Returns:
            bool: 路径是否有效
        """
        if not os.path.exists(path):
            return False
        
        # 检查必要的文件
        required_files = ['config.json']  # 至少需要 config.json
        
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                return False
        
        return True
    
    def load(self) -> VLMModelBase:
        """
        加载模型
        
        Returns:
            VLMModelBase: 加载的模型实例
            
        Raises:
            RuntimeError: 模型加载失败
            ValueError: 无效的模型配置
        """
        try:
            # 检查是否是本地路径
            is_local = os.path.exists(self.model_name_or_path)
            
            if is_local:
                logger.info(f"Loading model from local path: {self.model_name_or_path}")
                if not self._validate_model_path(self.model_name_or_path):
                    raise ValueError(
                        f"Invalid model path: {self.model_name_or_path}. "
                        "Required files (config.json) not found."
                    )
            else:
                logger.info(f"Loading model from HuggingFace: {self.model_name_or_path}")
            
            # 检测或使用指定的模型类型
            model_type = self.model_type or self._detect_model_type(self.model_name_or_path)
            logger.info(f"Detected model type: {model_type}")
            
            # 获取模型类
            model_class = self._get_model_class(model_type)
            logger.info(f"Using model class: {model_class.__name__}")
            
            # 实例化模型
            self.model_instance = model_class(
                model_name_or_path=self.model_name_or_path,
                device=self.device,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info("✓ Model loaded successfully")
            return self.model_instance
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def get_model(self) -> Optional[VLMModelBase]:
        """
        获取已加载的模型实例
        
        Returns:
            Optional[VLMModelBase]: 模型实例，如果未加载则返回 None
        """
        if self.model_instance is None:
            logger.warning("Model not loaded yet. Call load() first.")
        
        return self.model_instance
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        if self.model_instance is None:
            return {
                'status': 'not_loaded',
                'config': {
                    'model_name_or_path': self.model_name_or_path,
                    'device': self.device,
                    'dtype': str(self.torch_dtype)
                }
            }
        
        info = self.model_instance.get_model_info()
        info['status'] = 'loaded'
        return info
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, type]:
        """
        列出所有支持的模型类型
        
        Returns:
            Dict[str, type]: 模型类型映射
        """
        return cls.MODEL_MAPPING.copy()
    
    @classmethod
    def is_model_supported(cls, model_type: str) -> bool:
        """
        检查模型类型是否支持
        
        Args:
            model_type: 模型类型
            
        Returns:
            bool: 是否支持
        """
        return model_type.lower() in cls.MODEL_MAPPING
    
    def __repr__(self) -> str:
        status = "loaded" if self.model_instance else "not loaded"
        return (
            f"ModelLoader("
            f"model='{self.model_name_or_path}', "
            f"type='{self.model_type}', "
            f"device='{self.device}', "
            f"status='{status}')"
        )


def load_model(
    model_name_or_path: Optional[str] = None,
    model_type: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    **kwargs
) -> VLMModelBase:
    """
    快捷函数：加载模型
    
    Args:
        model_name_or_path: 模型名称或路径
        model_type: 模型类型
        device: 运行设备
        torch_dtype: 模型精度
        **kwargs: 其他参数
        
    Returns:
        VLMModelBase: 加载的模型实例
        
    Example:
        >>> model = load_model("Qwen/Qwen2-VL-7B-Instruct")
        >>> result = model.ocr(image, "识别图像中的文字")
    """
    loader = ModelLoader(
        model_name_or_path=model_name_or_path,
        model_type=model_type,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs
    )
    return loader.load()


class ModelFactory:
    """
    模型工厂
    统一的模型创建接口
    """
    
    # 模型注册表
    MODELS = {
        "qwen2-vl": Qwen2VLModel,
        "internvl": InternVLModel,
        "llava": LLaVAModel
    }
    
    # 模型别名
    MODEL_ALIASES = {
        "qwen": "qwen2-vl",
        "qwen2vl": "qwen2-vl",
        "llava-1.5": "llava",
        "llava-1.6": "llava",
        "internvl-2.5": "internvl"
    }
    
    @classmethod
    def create_model(
        cls, 
        model_type: str, 
        model_name: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ) -> VLMModelBase:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型 (qwen2-vl, internvl, llava)
            model_name: 模型名称或HuggingFace ID (可选)
            device: 运行设备
            **kwargs: 其他模型参数
            
        Returns:
            VLMModelBase: 模型实例
            
        Raises:
            ValueError: 不支持的模型类型
        """
        # 处理别名
        model_type = cls.MODEL_ALIASES.get(model_type, model_type)
        
        if model_type not in cls.MODELS:
            available = list(cls.MODELS.keys()) + list(cls.MODEL_ALIASES.keys())
            raise ValueError(
                f"不支持的模型类型: {model_type}。"
                f"可用模型: {', '.join(available)}"
            )
        
        model_class = cls.MODELS[model_type]
        
        # 创建模型实例
        if model_name:
            model_instance = model_class(model_name, device, **kwargs)
        else:
            model_instance = model_class(device=device, **kwargs)
        
        logger.info(f"Created {model_class.__name__} instance")
        
        return model_instance
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        列出所有支持的模型类型
        
        Returns:
            List[str]: 模型类型列表
        """
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict:
        """
        获取模型信息
        
        Args:
            model_type: 模型类型
            
        Returns:
            Dict: 模型信息
        """
        model_type = cls.MODEL_ALIASES.get(model_type, model_type)
        
        if model_type not in cls.MODELS:
            return {}
        
        model_class = cls.MODELS[model_type]
        
        # 获取模型类的基本信息
        info = {
            "type": model_type,
            "class": model_class.__name__,
            "description": model_class.__doc__ or "No description available"
        }
        
        # 如果有默认模型，添加到信息中
        if hasattr(model_class, 'DEFAULT_MODEL'):
            info["default_model"] = model_class.DEFAULT_MODEL
        
        if hasattr(model_class, 'SUPPORTED_MODELS'):
            info["supported_models"] = model_class.SUPPORTED_MODELS
        
        return info


class MultiModelLoader:
    """
    多模型加载器
    支持同时加载和管理多个模型
    """
    
    def __init__(self):
        """初始化多模型加载器"""
        self.models: Dict[str, VLMModelBase] = {}
        self.active_model: Optional[str] = None
        logger.info("MultiModelLoader initialized")
    
    def load(
        self, 
        model_name: str, 
        model_type: str,
        device: str = "cuda",
        **kwargs
    ) -> VLMModelBase:
        """
        加载模型
        
        Args:
            model_name: 模型标识名（用于管理）
            model_type: 模型类型
            device: 运行设备
            **kwargs: 其他模型参数
            
        Returns:
            VLMModelBase: 加载的模型实例
        """
        if model_name in self.models:
            logger.info(f"Model '{model_name}' already loaded")
            return self.models[model_name]
        
        logger.info(f"Loading model '{model_name}' (type: {model_type})...")
        
        # 创建模型实例
        model = ModelFactory.create_model(
            model_type, 
            device=device,
            **kwargs
        )
        
        # 保存到模型字典
        self.models[model_name] = model
        
        # 如果是第一个模型，设置为活动模型
        if self.active_model is None:
            self.active_model = model_name
        
        logger.info(f"Model '{model_name}' loaded successfully")
        
        return model
    
    def get(self, model_name: Optional[str] = None) -> Optional[VLMModelBase]:
        """
        获取已加载的模型
        
        Args:
            model_name: 模型名称（None表示获取活动模型）
            
        Returns:
            VLMModelBase: 模型实例，如果不存在返回None
        """
        if model_name is None:
            model_name = self.active_model
        
        return self.models.get(model_name)
    
    def set_active(self, model_name: str) -> bool:
        """
        设置活动模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功设置
        """
        if model_name not in self.models:
            logger.error(f"Model '{model_name}' not found")
            return False
        
        self.active_model = model_name
        logger.info(f"Active model set to '{model_name}'")
        return True
    
    def unload(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功卸载
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found")
            return False
        
        logger.info(f"Unloading model '{model_name}'...")
        
        # 删除模型
        del self.models[model_name]
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 如果卸载的是活动模型，切换到其他模型
        if self.active_model == model_name:
            if self.models:
                self.active_model = list(self.models.keys())[0]
                logger.info(f"Active model switched to '{self.active_model}'")
            else:
                self.active_model = None
        
        logger.info(f"Model '{model_name}' unloaded successfully")
        return True
    
    def list_loaded_models(self) -> List[str]:
        """
        列出所有已加载的模型
        
        Returns:
            List[str]: 模型名称列表
        """
        return list(self.models.keys())
    
    def get_model_status(self) -> Dict:
        """
        获取所有模型的状态
        
        Returns:
            Dict: 模型状态信息
        """
        status = {
            "active_model": self.active_model,
            "loaded_models": {},
            "total_models": len(self.models)
        }
        
        for name, model in self.models.items():
            status["loaded_models"][name] = {
                "type": model.__class__.__name__,
                "device": model.device,
                "is_active": name == self.active_model
            }
        
        return status
    
    def clear_all(self):
        """卸载所有模型"""
        logger.info("Clearing all models...")
        
        model_names = list(self.models.keys())
        for name in model_names:
            self.unload(name)
        
        logger.info("All models cleared")
    
    def __repr__(self) -> str:
        return (
            f"MultiModelLoader("
            f"loaded={len(self.models)}, "
            f"active='{self.active_model}')"
        )
