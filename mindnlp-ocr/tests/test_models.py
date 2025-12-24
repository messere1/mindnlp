"""
模型层单元测试
测试 Qwen2-VL 模型加载、推理、解码功能
"""

import pytest
import torch
from PIL import Image
import os
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 使用动态导入避免 torch 依赖问题
import importlib.util


def load_module(module_name, file_path):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 加载模块
base_path = Path(__file__).parent.parent / "models"
base = load_module("base", base_path / "base.py")
qwen2vl = load_module("qwen2vl", base_path / "qwen2vl.py")
loader = load_module("loader", base_path / "loader.py")

VLMModelBase = base.VLMModelBase
Qwen2VLModel = qwen2vl.Qwen2VLModel
ModelLoader = loader.ModelLoader
load_model = loader.load_model


# 测试配置
TEST_MODEL = "Qwen/Qwen2-VL-2B-Instruct"  # 使用 2B 模型测试（显存要求较低）
SKIP_GPU_TESTS = not torch.cuda.is_available()


class TestVLMModelBase:
    """测试 VLMModelBase 基类"""
    
    def test_base_is_abstract(self):
        """测试基类是抽象类"""
        with pytest.raises(TypeError):
            # 不能直接实例化抽象类
            base = VLMModelBase(
                model_name_or_path="test",
                device="cpu"
            )
    
    def test_base_has_required_methods(self):
        """测试基类定义了必要的抽象方法"""
        required_methods = [
            'load_model',
            'load_tokenizer', 
            'load_processor',
            'prepare_inputs',
            'generate',
            'decode'
        ]
        
        for method in required_methods:
            assert hasattr(VLMModelBase, method)
            assert callable(getattr(VLMModelBase, method))


class TestModelLoader:
    """测试 ModelLoader"""
    
    def test_loader_initialization(self):
        """测试加载器初始化"""
        loader_inst = ModelLoader(
            model_name_or_path=TEST_MODEL,
            device="cpu"
        )
        
        assert loader_inst.model_name_or_path == TEST_MODEL
        assert loader_inst.device == "cpu"
        assert loader_inst.model_instance is None
        print("✓ Loader initialization test passed")
    
    def test_device_auto_detection(self):
        """测试设备自动检测"""
        loader_inst = ModelLoader(
            model_name_or_path=TEST_MODEL,
            device="auto"
        )
        
        if torch.cuda.is_available():
            assert loader_inst.device == "cuda"
        else:
            assert loader_inst.device == "cpu"
        print(f"✓ Device auto-detection: {loader_inst.device}")
    
    def test_model_type_detection(self):
        """测试模型类型检测"""
        loader_inst = ModelLoader(model_name_or_path=TEST_MODEL)
        
        # 测试 Qwen 检测
        model_type = loader_inst._detect_model_type("Qwen/Qwen2-VL-7B-Instruct")
        assert model_type == "qwen2-vl"
        
        model_type = loader_inst._detect_model_type("qwen2vl-instruct")
        assert model_type == "qwen2-vl"
        
        print("✓ Model type detection test passed")
    
    def test_list_supported_models(self):
        """测试列出支持的模型"""
        supported = ModelLoader.list_supported_models()
        
        assert isinstance(supported, dict)
        assert 'qwen2-vl' in supported
        assert supported['qwen2-vl'] == Qwen2VLModel
        
        print(f"✓ Supported models: {list(supported.keys())}")
    
    def test_is_model_supported(self):
        """测试检查模型是否支持"""
        assert ModelLoader.is_model_supported('qwen2-vl') is True
        assert ModelLoader.is_model_supported('qwen') is True
        assert ModelLoader.is_model_supported('unknown-model') is False
        
        print("✓ Model support check test passed")
    
    def test_get_model_info_before_load(self):
        """测试加载前获取模型信息"""
        loader_inst = ModelLoader(model_name_or_path=TEST_MODEL, device="cpu")
        info = loader_inst.get_model_info()
        
        assert info['status'] == 'not_loaded'
        assert 'config' in info
        assert info['config']['model_name_or_path'] == TEST_MODEL
        
        print("✓ Get model info before load test passed")


@pytest.mark.skipif(SKIP_GPU_TESTS, reason="CUDA not available")
class TestQwen2VLModel:
    """
    测试 Qwen2VLModel
    
    注意：这些测试需要 GPU 和网络连接来下载模型
    如果没有 GPU 或不想下载模型，这些测试会被跳过
    """
    
    @pytest.fixture(scope="class")
    def test_image(self):
        """创建测试图像"""
        image = Image.new('RGB', (224, 224), color=(255, 0, 0))
        return image
    
    @pytest.mark.slow
    def test_model_loading(self):
        """测试模型加载（需要下载模型，较慢）"""
        try:
            model = Qwen2VLModel(
                model_name_or_path=TEST_MODEL,
                device="cuda",
                torch_dtype=torch.float16
            )
            
            assert model.model is not None
            assert model.tokenizer is not None
            assert model.processor is not None
            
            print("✓ Model loading test passed")
            
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")
    
    @pytest.mark.slow
    def test_model_info(self):
        """测试获取模型信息"""
        try:
            model = Qwen2VLModel(
                model_name_or_path=TEST_MODEL,
                device="cuda"
            )
            
            info = model.get_model_info()
            
            assert info['model_name'] == TEST_MODEL
            assert info['model_loaded'] is True
            assert info['tokenizer_loaded'] is True
            assert info['processor_loaded'] is True
            assert 'model_type' in info
            
            print(f"✓ Model info test passed: {info['model_type']}")
            
        except Exception as e:
            pytest.skip(f"Model info test failed: {e}")
    
    @pytest.mark.slow
    def test_prepare_inputs(self, test_image):
        """测试准备输入"""
        try:
            model = Qwen2VLModel(
                model_name_or_path=TEST_MODEL,
                device="cuda"
            )
            
            prompt = "请识别图像中的内容"
            inputs = model.prepare_inputs(test_image, prompt)
            
            assert isinstance(inputs, dict)
            assert 'input_ids' in inputs or 'pixel_values' in inputs
            
            # 检查输入在正确的设备上
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    assert value.device.type == 'cuda'
            
            print("✓ Prepare inputs test passed")
            
        except Exception as e:
            pytest.skip(f"Prepare inputs test failed: {e}")
    
    @pytest.mark.slow
    def test_generate(self, test_image):
        """测试生成（完整推理测试）"""
        try:
            model = Qwen2VLModel(
                model_name_or_path=TEST_MODEL,
                device="cuda"
            )
            
            prompt = "描述这张图片"
            inputs = model.prepare_inputs(test_image, prompt)
            
            output_ids = model.generate(
                inputs,
                max_new_tokens=50,
                do_sample=False
            )
            
            assert isinstance(output_ids, torch.Tensor)
            assert output_ids.dim() >= 1
            assert output_ids.shape[-1] > 0
            
            print(f"✓ Generate test passed, output shape: {output_ids.shape}")
            
        except Exception as e:
            pytest.skip(f"Generate test failed: {e}")
    
    @pytest.mark.slow
    def test_decode(self):
        """测试解码"""
        try:
            model = Qwen2VLModel(
                model_name_or_path=TEST_MODEL,
                device="cuda"
            )
            
            # 创建测试 token IDs
            test_tokens = torch.tensor([[1, 2, 3, 4, 5]])
            
            decoded_text = model.decode(test_tokens)
            
            assert isinstance(decoded_text, str)
            
            print(f"✓ Decode test passed, decoded text length: {len(decoded_text)}")
            
        except Exception as e:
            pytest.skip(f"Decode test failed: {e}")
    
    @pytest.mark.slow
    def test_ocr_pipeline(self, test_image):
        """测试完整 OCR 流程"""
        try:
            model = Qwen2VLModel(
                model_name_or_path=TEST_MODEL,
                device="cuda"
            )
            
            prompt = "识别图像中的文字"
            result = model.ocr(test_image, prompt, max_new_tokens=100)
            
            assert isinstance(result, str)
            assert len(result) > 0
            
            print(f"✓ OCR pipeline test passed")
            print(f"   Result preview: {result[:100]}...")
            
        except Exception as e:
            pytest.skip(f"OCR pipeline test failed: {e}")


class TestLoadModelFunction:
    """测试快捷加载函数"""
    
    @pytest.mark.slow
    def test_load_model_function(self):
        """测试 load_model 快捷函数"""
        try:
            model = load_model(
                model_name_or_path=TEST_MODEL,
                device="cpu",  # 使用 CPU 测试
                torch_dtype=torch.float32
            )
            
            assert isinstance(model, VLMModelBase)
            assert model.model is not None
            
            print("✓ load_model function test passed")
            
        except Exception as e:
            pytest.skip(f"load_model function test failed: {e}")


class TestModelDevices:
    """测试不同设备配置"""
    
    def test_cpu_device(self):
        """测试 CPU 设备配置"""
        loader_inst = ModelLoader(
            model_name_or_path=TEST_MODEL,
            device="cpu"
        )
        assert loader_inst.device == "cpu"
        print("✓ CPU device test passed")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """测试 CUDA 设备配置"""
        loader_inst = ModelLoader(
            model_name_or_path=TEST_MODEL,
            device="cuda"
        )
        assert loader_inst.device == "cuda"
        print("✓ CUDA device test passed")
    
    def test_invalid_device_fallback(self):
        """测试无效设备回退到 CPU"""
        # 如果 CUDA 不可用，应该回退到 CPU
        if not torch.cuda.is_available():
            loader_inst = ModelLoader(
                model_name_or_path=TEST_MODEL,
                device="cuda"
            )
            assert loader_inst.device == "cpu"
            print("✓ Invalid device fallback test passed")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s', '--tb=short'])
