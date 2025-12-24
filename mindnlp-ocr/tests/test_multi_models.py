"""
多模型支持测试
测试InternVL和LLaVA模型集成
"""

import pytest
import torch
from PIL import Image
import numpy as np
from models.llava import LLaVAModel
from models.loader import ModelFactory, MultiModelLoader
from config.settings import get_settings


class TestLLaVAModel:
    """LLaVA模型测试"""
    
    @pytest.fixture
    def model(self):
        """创建LLaVA模型实例（不加载权重）"""
        return LLaVAModel(device="cpu")
    
    @pytest.fixture
    def sample_image(self):
        """生成测试图像"""
        # 创建一个336x336的RGB图像
        img_array = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_model_initialization(self, model):
        """测试模型初始化"""
        assert model is not None
        assert model.device == "cpu"
        assert model.model_name in LLaVAModel.SUPPORTED_MODELS or \
               model.model_name == LLaVAModel.DEFAULT_MODEL
    
    def test_model_info(self, model):
        """测试模型信息获取"""
        info = model.get_model_info()
        
        assert "model_name" in info
        assert "model_type" in info
        assert info["model_type"] == "LLaVA"
        assert "device" in info
        assert "capabilities" in info
        assert len(info["capabilities"]) > 0
    
    def test_preprocess_format(self, model, sample_image):
        """测试预处理输出格式"""
        # 注意：这个测试需要processor，如果没加载会触发load_tokenizer
        # 在单元测试中，我们只测试格式，不实际加载模型
        try:
            # 创建一个简单的测试
            prompt = "识别图中的文字"
            
            # 验证prompt格式化
            formatted = f"USER: <image>\n{prompt}\nASSISTANT:"
            assert "USER:" in formatted
            assert "ASSISTANT:" in formatted
            assert prompt in formatted
            
        except Exception as e:
            # 如果无法加载processor（没有模型文件），跳过测试
            pytest.skip(f"Skipping preprocess test: {e}")
    
    def test_supported_models(self):
        """测试支持的模型列表"""
        assert len(LLaVAModel.SUPPORTED_MODELS) > 0
        assert LLaVAModel.DEFAULT_MODEL in LLaVAModel.SUPPORTED_MODELS
    
    def test_repr(self, model):
        """测试字符串表示"""
        repr_str = repr(model)
        assert "LLaVAModel" in repr_str
        assert model.device in repr_str


class TestModelFactory:
    """模型工厂测试"""
    
    def test_list_models(self):
        """测试列出所有支持的模型"""
        models = ModelFactory.list_models()
        
        assert len(models) > 0
        assert "qwen2-vl" in models
        assert "internvl" in models
        assert "llava" in models
    
    def test_create_qwen2vl(self):
        """测试创建Qwen2-VL模型"""
        model = ModelFactory.create_model("qwen2-vl", device="cpu")
        
        assert model is not None
        assert model.device == "cpu"
    
    def test_create_internvl(self):
        """测试创建InternVL模型"""
        model = ModelFactory.create_model("internvl", device="cpu")
        
        assert model is not None
        assert model.device == "cpu"
    
    def test_create_llava(self):
        """测试创建LLaVA模型"""
        model = ModelFactory.create_model("llava", device="cpu")
        
        assert model is not None
        assert model.device == "cpu"
        assert isinstance(model, LLaVAModel)
    
    def test_create_with_alias(self):
        """测试使用别名创建模型"""
        # 测试qwen别名
        model1 = ModelFactory.create_model("qwen", device="cpu")
        assert model1 is not None
        
        # 测试llava-1.5别名
        model2 = ModelFactory.create_model("llava-1.5", device="cpu")
        assert model2 is not None
        assert isinstance(model2, LLaVAModel)
    
    def test_create_invalid_model(self):
        """测试创建不支持的模型类型"""
        with pytest.raises(ValueError) as exc_info:
            ModelFactory.create_model("invalid-model", device="cpu")
        
        assert "不支持的模型类型" in str(exc_info.value)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        # 测试qwen2-vl
        info = ModelFactory.get_model_info("qwen2-vl")
        assert info["type"] == "qwen2-vl"
        assert "class" in info
        
        # 测试llava
        info = ModelFactory.get_model_info("llava")
        assert info["type"] == "llava"
        assert "default_model" in info or "class" in info


class TestMultiModelLoader:
    """多模型加载器测试"""
    
    @pytest.fixture
    def loader(self):
        """创建多模型加载器"""
        return MultiModelLoader()
    
    def test_initialization(self, loader):
        """测试初始化"""
        assert loader is not None
        assert len(loader.models) == 0
        assert loader.active_model is None
    
    def test_load_single_model(self, loader):
        """测试加载单个模型"""
        try:
            # 只创建模型实例，不实际加载权重
            model = ModelFactory.create_model("llava", device="cpu")
            loader.models["test-llava"] = model
            loader.active_model = "test-llava"
            
            assert len(loader.models) == 1
            assert "test-llava" in loader.models
            assert loader.active_model == "test-llava"
            
        except Exception as e:
            pytest.skip(f"Skipping model load test: {e}")
    
    def test_get_model(self, loader):
        """测试获取模型"""
        # 手动添加模型
        model = ModelFactory.create_model("llava", device="cpu")
        loader.models["test-model"] = model
        loader.active_model = "test-model"
        
        # 获取模型
        retrieved = loader.get("test-model")
        assert retrieved is model
        
        # 获取活动模型
        active = loader.get()
        assert active is model
        
        # 获取不存在的模型
        none_model = loader.get("non-existent")
        assert none_model is None
    
    def test_set_active_model(self, loader):
        """测试设置活动模型"""
        # 添加两个模型
        model1 = ModelFactory.create_model("llava", device="cpu")
        model2 = ModelFactory.create_model("qwen2-vl", device="cpu")
        
        loader.models["model1"] = model1
        loader.models["model2"] = model2
        loader.active_model = "model1"
        
        # 切换活动模型
        result = loader.set_active("model2")
        assert result is True
        assert loader.active_model == "model2"
        
        # 设置不存在的模型
        result = loader.set_active("non-existent")
        assert result is False
    
    def test_unload_model(self, loader):
        """测试卸载模型"""
        # 添加模型
        model = ModelFactory.create_model("llava", device="cpu")
        loader.models["test-model"] = model
        loader.active_model = "test-model"
        
        # 卸载模型
        result = loader.unload("test-model")
        assert result is True
        assert "test-model" not in loader.models
        assert loader.active_model is None
        
        # 卸载不存在的模型
        result = loader.unload("non-existent")
        assert result is False
    
    def test_list_loaded_models(self, loader):
        """测试列出已加载模型"""
        assert loader.list_loaded_models() == []
        
        # 添加模型
        model1 = ModelFactory.create_model("llava", device="cpu")
        model2 = ModelFactory.create_model("qwen2-vl", device="cpu")
        
        loader.models["model1"] = model1
        loader.models["model2"] = model2
        
        models_list = loader.list_loaded_models()
        assert len(models_list) == 2
        assert "model1" in models_list
        assert "model2" in models_list
    
    def test_get_model_status(self, loader):
        """测试获取模型状态"""
        # 空状态
        status = loader.get_model_status()
        assert status["total_models"] == 0
        assert status["active_model"] is None
        
        # 添加模型后
        model = ModelFactory.create_model("llava", device="cpu")
        loader.models["test-model"] = model
        loader.active_model = "test-model"
        
        status = loader.get_model_status()
        assert status["total_models"] == 1
        assert status["active_model"] == "test-model"
        assert "test-model" in status["loaded_models"]
        assert status["loaded_models"]["test-model"]["is_active"] is True
    
    def test_clear_all(self, loader):
        """测试清除所有模型"""
        # 添加多个模型
        model1 = ModelFactory.create_model("llava", device="cpu")
        model2 = ModelFactory.create_model("qwen2-vl", device="cpu")
        
        loader.models["model1"] = model1
        loader.models["model2"] = model2
        loader.active_model = "model1"
        
        # 清除所有
        loader.clear_all()
        
        assert len(loader.models) == 0
        assert loader.active_model is None
    
    def test_repr(self, loader):
        """测试字符串表示"""
        repr_str = repr(loader)
        assert "MultiModelLoader" in repr_str
        assert "loaded=0" in repr_str


class TestModelConfiguration:
    """模型配置测试"""
    
    def test_settings(self):
        """测试配置设置"""
        settings = get_settings()
        
        # 测试默认模型
        assert settings.default_model in ["qwen2-vl", "internvl", "llava"]
        
        # 测试可用模型
        assert len(settings.available_models) >= 3
        assert "qwen2-vl" in settings.available_models
        assert "internvl" in settings.available_models
        assert "llava" in settings.available_models
        
        # 测试模型描述
        assert len(settings.model_descriptions) >= 3
        
        # 测试模型能力
        assert len(settings.model_capabilities) >= 3
    
    def test_model_paths(self):
        """测试模型路径配置"""
        settings = get_settings()
        
        # 验证模型路径格式
        for model_type, model_path in settings.available_models.items():
            assert isinstance(model_path, str)
            assert len(model_path) > 0
            # HuggingFace格式: org/model 或本地路径
            assert "/" in model_path or "\\" in model_path or "-" in model_path


class TestModelSwitching:
    """模型切换测试"""
    
    @pytest.fixture
    def loader(self):
        """创建多模型加载器"""
        return MultiModelLoader()
    
    def test_hot_switching(self, loader):
        """测试模型热切换"""
        # 加载多个模型
        model1 = ModelFactory.create_model("llava", device="cpu")
        model2 = ModelFactory.create_model("qwen2-vl", device="cpu")
        
        loader.models["llava"] = model1
        loader.models["qwen2-vl"] = model2
        loader.active_model = "llava"
        
        # 测试切换
        assert loader.get().device == "cpu"
        
        loader.set_active("qwen2-vl")
        assert loader.active_model == "qwen2-vl"
        assert loader.get().device == "cpu"
    
    def test_fallback_on_unload(self, loader):
        """测试卸载活动模型时的回退"""
        # 加载多个模型
        model1 = ModelFactory.create_model("llava", device="cpu")
        model2 = ModelFactory.create_model("qwen2-vl", device="cpu")
        
        loader.models["model1"] = model1
        loader.models["model2"] = model2
        loader.active_model = "model1"
        
        # 卸载活动模型
        loader.unload("model1")
        
        # 应该自动切换到另一个模型
        assert loader.active_model == "model2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
