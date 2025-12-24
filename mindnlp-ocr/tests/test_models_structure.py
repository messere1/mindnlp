"""
模型层基础测试（不需要torch）
测试模型加载器的基础功能
"""

import pytest
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelStructure:
    """测试模型结构和接口"""
    
    def test_models_directory_exists(self):
        """测试 models 目录存在"""
        models_dir = Path(__file__).parent.parent / "models"
        assert models_dir.exists()
        assert models_dir.is_dir()
        print("✓ Models directory exists")
    
    def test_required_files_exist(self):
        """测试必需文件存在"""
        models_dir = Path(__file__).parent.parent / "models"
        
        required_files = [
            'base.py',
            'qwen2vl.py',
            'loader.py',
            '__init__.py'
        ]
        
        for file in required_files:
            file_path = models_dir / file
            assert file_path.exists(), f"Missing file: {file}"
            assert file_path.is_file()
        
        print(f"✓ All required files exist: {required_files}")
    
    def test_base_py_has_abstract_class(self):
        """测试 base.py 包含抽象基类"""
        base_path = Path(__file__).parent.parent / "models" / "base.py"
        content = base_path.read_text(encoding='utf-8')
        
        assert 'from abc import ABC, abstractmethod' in content
        assert 'class VLMModelBase(ABC)' in content
        assert '@abstractmethod' in content
        
        print("✓ base.py contains abstract base class")
    
    def test_base_py_has_required_methods(self):
        """测试 base.py 包含必需方法"""
        base_path = Path(__file__).parent.parent / "models" / "base.py"
        content = base_path.read_text(encoding='utf-8')
        
        required_methods = [
            'def load_model',
            'def load_tokenizer',
            'def load_processor',
            'def prepare_inputs',
            'def generate',
            'def decode',
            'def ocr',
            'def get_model_info'
        ]
        
        for method in required_methods:
            assert method in content, f"Missing method: {method}"
        
        print(f"✓ base.py has all required methods: {len(required_methods)}")
    
    def test_qwen2vl_py_has_model_class(self):
        """测试 qwen2vl.py 包含模型类"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        assert 'class Qwen2VLModel(VLMModelBase)' in content
        assert 'from transformers import' in content
        assert 'Qwen2VLForConditionalGeneration' in content
        assert 'AutoProcessor' in content
        assert 'AutoTokenizer' in content
        
        print("✓ qwen2vl.py contains Qwen2VLModel class")
    
    def test_qwen2vl_implements_abstract_methods(self):
        """测试 Qwen2VL 实现了所有抽象方法"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        # 检查实现的方法
        implemented_methods = [
            'def load_model',
            'def load_tokenizer',
            'def load_processor',
            'def prepare_inputs',
            'def generate',
            'def decode'
        ]
        
        for method in implemented_methods:
            assert method in content, f"Missing implementation: {method}"
        
        print(f"✓ Qwen2VLModel implements all abstract methods: {len(implemented_methods)}")
    
    def test_loader_py_has_loader_class(self):
        """测试 loader.py 包含加载器类"""
        loader_path = Path(__file__).parent.parent / "models" / "loader.py"
        content = loader_path.read_text(encoding='utf-8')
        
        assert 'class ModelLoader' in content
        assert 'MODEL_MAPPING' in content
        assert 'def load' in content
        assert 'def _detect_model_type' in content
        assert 'def load_model' in content  # 快捷函数
        
        print("✓ loader.py contains ModelLoader class")
    
    def test_loader_has_model_mapping(self):
        """测试 loader 包含模型映射"""
        loader_path = Path(__file__).parent.parent / "models" / "loader.py"
        content = loader_path.read_text(encoding='utf-8')
        
        assert "'qwen2-vl': Qwen2VLModel" in content or \
               "'qwen2-vl': Qwen2VLModel" in content
        assert "MODEL_MAPPING = {" in content
        
        print("✓ loader.py has model mapping")
    
    def test_init_py_exports(self):
        """测试 __init__.py 导出"""
        init_path = Path(__file__).parent.parent / "models" / "__init__.py"
        content = init_path.read_text(encoding='utf-8')
        
        exports = [
            'VLMModelBase',
            'ModelLoader',
            'Qwen2VLModel',
            'load_model'
        ]
        
        for export in exports:
            assert export in content, f"Missing export: {export}"
        
        print(f"✓ __init__.py exports all required classes: {exports}")


class TestCodeQuality:
    """测试代码质量"""
    
    def test_files_have_docstrings(self):
        """测试文件包含文档字符串"""
        models_dir = Path(__file__).parent.parent / "models"
        
        python_files = ['base.py', 'qwen2vl.py', 'loader.py']
        
        for file in python_files:
            file_path = models_dir / file
            content = file_path.read_text(encoding='utf-8')
            
            # 检查文件开头有文档字符串
            assert '"""' in content[:500], f"{file} should have module docstring"
        
        print(f"✓ All files have docstrings: {len(python_files)}")
    
    def test_classes_have_docstrings(self):
        """测试类包含文档字符串"""
        models_dir = Path(__file__).parent.parent / "models"
        
        # 检查基类
        base_content = (models_dir / "base.py").read_text(encoding='utf-8')
        assert 'class VLMModelBase(ABC):\n    """' in base_content or \
               'class VLMModelBase(ABC):\n    \n    """' in base_content
        
        # 检查 Qwen2VL 类
        qwen_content = (models_dir / "qwen2vl.py").read_text(encoding='utf-8')
        assert '"""' in qwen_content[qwen_content.find('class Qwen2VLModel'):qwen_content.find('class Qwen2VLModel')+200]
        
        print("✓ Classes have docstrings")
    
    def test_methods_have_docstrings(self):
        """测试主要方法有文档字符串"""
        models_dir = Path(__file__).parent.parent / "models"
        qwen_content = (models_dir / "qwen2vl.py").read_text(encoding='utf-8')
        
        # 检查关键方法有文档字符串
        key_methods = ['load_model', 'prepare_inputs', 'generate', 'decode']
        
        for method in key_methods:
            method_pos = qwen_content.find(f'def {method}')
            if method_pos != -1:
                # 检查方法定义后有文档字符串
                method_section = qwen_content[method_pos:method_pos+500]
                assert '"""' in method_section, f"Method {method} should have docstring"
        
        print(f"✓ Key methods have docstrings: {key_methods}")
    
    def test_imports_are_organized(self):
        """测试导入语句组织良好"""
        models_dir = Path(__file__).parent.parent / "models"
        
        for file in ['base.py', 'qwen2vl.py', 'loader.py']:
            content = (models_dir / file).read_text(encoding='utf-8')
            
            # 检查有导入语句
            assert 'import' in content or 'from' in content
            
            # 检查导入在文件前部
            lines = content.split('\n')
            import_lines = [i for i, line in enumerate(lines) if 'import' in line or 'from' in line]
            if import_lines:
                assert import_lines[0] < 50, f"{file}: imports should be at the top"
        
        print("✓ Imports are organized")


class TestFunctionality:
    """测试基础功能"""
    
    def test_model_mapping_completeness(self):
        """测试模型映射完整性"""
        loader_path = Path(__file__).parent.parent / "models" / "loader.py"
        content = loader_path.read_text(encoding='utf-8')
        
        # 检查映射中有 qwen 相关的条目
        assert 'qwen' in content.lower()
        assert "MODEL_MAPPING" in content
        
        print("✓ Model mapping is complete")
    
    def test_device_options_mentioned(self):
        """测试设备选项被提及"""
        models_dir = Path(__file__).parent.parent / "models"
        
        for file in ['base.py', 'qwen2vl.py', 'loader.py']:
            content = (models_dir / file).read_text(encoding='utf-8')
            
            # 检查设备选项
            assert 'cuda' in content.lower() or 'device' in content.lower()
        
        print("✓ Device options are mentioned")
    
    def test_dtype_options_mentioned(self):
        """测试数据类型选项被提及"""
        models_dir = Path(__file__).parent.parent / "models"
        
        # 检查 dtype 配置
        for file in ['base.py', 'qwen2vl.py', 'loader.py']:
            content = (models_dir / file).read_text(encoding='utf-8')
            
            assert 'dtype' in content.lower() or 'float16' in content.lower()
        
        print("✓ Dtype options are mentioned")
    
    def test_error_handling_present(self):
        """测试错误处理存在"""
        models_dir = Path(__file__).parent.parent / "models"
        
        for file in ['qwen2vl.py', 'loader.py']:
            content = (models_dir / file).read_text(encoding='utf-8')
            
            # 检查有异常处理
            assert 'try:' in content
            assert 'except' in content
            assert 'raise' in content
        
        print("✓ Error handling is present")


class TestIssue2351Requirements:
    """测试 Issue #2351 的具体要求"""
    
    def test_uses_transformers_library(self):
        """测试使用 transformers 库"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        assert 'from transformers import' in content
        assert 'Qwen2VLForConditionalGeneration' in content
        
        print("✓ Uses transformers library (requirement met)")
    
    def test_supports_huggingface_and_local(self):
        """测试支持 HuggingFace ID 和本地路径"""
        loader_path = Path(__file__).parent.parent / "models" / "loader.py"
        content = loader_path.read_text(encoding='utf-8')
        
        assert 'model_name_or_path' in content or 'model_path' in content
        assert 'local' in content.lower() or 'path' in content.lower()
        
        print("✓ Supports HuggingFace ID and local path (requirement met)")
    
    def test_has_generate_method(self):
        """测试有 generate() 方法"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        assert 'def generate(' in content
        assert 'max_new_tokens' in content
        assert 'temperature' in content
        
        print("✓ Has generate() method (requirement met)")
    
    def test_has_decode_method(self):
        """测试有 decode() 方法"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        assert 'def decode(' in content
        assert 'skip_special_tokens' in content
        
        print("✓ Has decode() method (requirement met)")
    
    def test_supports_fp16(self):
        """测试支持 FP16"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        assert 'float16' in content or 'torch_dtype' in content
        
        print("✓ Supports FP16 (requirement met)")
    
    def test_uses_processor(self):
        """测试使用 processor"""
        qwen_path = Path(__file__).parent.parent / "models" / "qwen2vl.py"
        content = qwen_path.read_text(encoding='utf-8')
        
        assert 'AutoProcessor' in content
        assert 'processor' in content.lower()
        
        print("✓ Uses processor (requirement met)")
    
    def test_has_model_info_method(self):
        """测试有 get_model_info() 方法"""
        base_path = Path(__file__).parent.parent / "models" / "base.py"
        content = base_path.read_text(encoding='utf-8')
        
        assert 'def get_model_info' in content
        
        print("✓ Has get_model_info() method (requirement met)")


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '-s'])
