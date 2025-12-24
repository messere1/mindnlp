"""
配置设置
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from functools import lru_cache


class Settings(BaseModel):
    """应用配置"""
    
    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # 模型配置
    default_model: str = "qwen2-vl"
    default_model_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cuda"
    
    # 多模型支持
    enable_multi_model: bool = Field(
        default=False,
        description="是否启用多模型同时加载"
    )
    
    available_models: Dict[str, str] = Field(
        default={
            "qwen2-vl": "Qwen/Qwen2-VL-2B-Instruct",
            "internvl": "OpenGVLab/InternVL2_5-8B",
            "llava": "llava-hf/llava-1.5-7b-hf"
        },
        description="可用的模型列表 {model_type: model_path}"
    )
    
    model_descriptions: Dict[str, str] = Field(
        default={
            "qwen2-vl": "Qwen2-VL: 强大的多语言多模态模型，支持动态分辨率",
            "internvl": "InternVL: 多模态理解能力强，适合文档分析",
            "llava": "LLaVA: 视觉指令遵循模型，推理速度快"
        },
        description="模型描述"
    )
    
    model_capabilities: Dict[str, List[str]] = Field(
        default={
            "qwen2-vl": ["OCR", "文档分析", "多语言", "高精度"],
            "internvl": ["文档理解", "表格识别", "复杂布局"],
            "llava": ["快速识别", "指令遵循", "简单OCR"]
        },
        description="模型能力标签"
    )
    
    # 图像处理配置
    max_image_size: int = 10 * 1024 * 1024  # 10MB
    target_image_size: tuple = (448, 448)
    
    # 推理配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 日志配置
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "OCR_"


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例
    
    Returns:
        Settings: 配置对象
    """
    return Settings()
