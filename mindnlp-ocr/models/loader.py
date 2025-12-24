"""
模型加载器
根据模型名称加载对应的VLM模型
支持多模型管理和动态切换
"""

import torch
from typing import Dict, Optional, List, Union
from .base import VLMModelBase
from .qwen2vl import Qwen2VLModel
from .internvl import InternVLModel
from .llava import LLaVAModel
from utils.logger import get_logger


logger = get_logger(__name__)


class ModelLoader:
    """模型加载器"""
    
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
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化模型加载器
        
        Args:
            model_name: 模型名称或HuggingFace model ID
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model_instance = None
        logger.info(f"ModelLoader initialized with model: {model_name}")
    
    def load_model(self) -> VLMModelBase:
        """
        加载模型
        
        Returns:
            VLMModelBase: 加载的模型实例
        """
        # 检测模型类型
        model_type = self._detect_model_type(self.model_name)
        
        # 获取对应的模型类
        model_class = self.MODEL_MAPPING.get(model_type)
        
        if model_class is None:
            logger.warning(f"Unknown model type: {model_type}, using Qwen2VL as default")
            model_class = Qwen2VLModel
        
        # 实例化并加载模型
        logger.info(f"Loading model with {model_class.__name__}")
        self.model_instance = model_class(self.model_name, self.device)
        
        return self.model_instance.model
    
    def load_tokenizer(self):
        """
        加载tokenizer
        
        Returns:
            Tokenizer实例
        """
        if self.model_instance is None:
            self.load_model()
        
        return self.model_instance.tokenizer
    
    def _detect_model_type(self, model_name: str) -> str:
        """
        检测模型类型
        
        Args:
            model_name: 模型名称
            
        Returns:
            str: 模型类型
        """
        model_name_lower = model_name.lower()
        
        if 'qwen' in model_name_lower:
            return 'qwen2-vl'
        elif 'internvl' in model_name_lower:
            return 'internvl'
        elif 'llava' in model_name_lower:
            return 'llava'
        else:
            logger.warning(f"Cannot detect model type from name: {model_name}")
            return 'qwen2-vl'  # 默认使用Qwen2-VL


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
        
        # 加载模型权重
        model.load_model()
        model.load_tokenizer()
        
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

