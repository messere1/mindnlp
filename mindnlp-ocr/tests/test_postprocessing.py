"""
后处理组件测试
"""

import pytest
import json
from unittest.mock import Mock
from core.parser import (
    TokenDecoder,
    ResultParser,
    OutputFormatter,
    OCRResult,
    OCRTextBlock
)


class TestOCRTextBlock:
    """测试 OCRTextBlock 数据类"""
    
    def test_create_text_block(self):
        """测试创建文本块"""
        block = OCRTextBlock(
            text="Hello World",
            bbox=[10, 20, 100, 50],
            confidence=0.95
        )
        
        assert block.text == "Hello World"
        assert block.bbox == [10, 20, 100, 50]
        assert block.confidence == 0.95
    
    def test_text_block_properties(self):
        """测试文本块属性"""
        block = OCRTextBlock(
            text="Test",
            bbox=[10, 20, 110, 70]
        )
        
        assert block.x1 == 10
        assert block.y1 == 20
        assert block.x2 == 110
        assert block.y2 == 70
        assert block.width == 100
        assert block.height == 50
        assert block.area == 5000
        assert block.center == (60, 45)
    
    def test_text_block_validation(self):
        """测试文本块验证"""
        # 无效的 bbox
        with pytest.raises(ValueError):
            OCRTextBlock(text="Test", bbox=[10, 20, 30])
        
        # 无效的 confidence
        with pytest.raises(ValueError):
            OCRTextBlock(text="Test", bbox=[10, 20, 30, 40], confidence=1.5)
    
    def test_text_block_to_dict(self):
        """测试转换为字典"""
        block = OCRTextBlock(
            text="Test",
            bbox=[10, 20, 30, 40],
            confidence=0.9,
            language="en"
        )
        
        data = block.to_dict()
        assert data['text'] == "Test"
        assert data['bbox'] == [10, 20, 30, 40]
        assert data['confidence'] == 0.9
        assert data['language'] == "en"
    
    def test_text_block_from_dict(self):
        """测试从字典创建"""
        data = {
            'text': "Test",
            'bbox': [10, 20, 30, 40],
            'confidence': 0.9
        }
        
        block = OCRTextBlock.from_dict(data)
        assert block.text == "Test"
        assert block.bbox == [10, 20, 30, 40]
        assert block.confidence == 0.9


class TestOCRResult:
    """测试 OCRResult 数据类"""
    
    def test_create_result(self):
        """测试创建 OCR 结果"""
        blocks = [
            OCRTextBlock(text="Hello", bbox=[0, 0, 50, 20]),
            OCRTextBlock(text="World", bbox=[60, 0, 120, 20])
        ]
        
        result = OCRResult(text_blocks=blocks)
        assert len(result) == 2
        assert result.full_text == "Hello\nWorld"
    
    def test_result_properties(self):
        """测试结果属性"""
        blocks = [
            OCRTextBlock(text="Test", bbox=[0, 0, 50, 20], confidence=0.9),
            OCRTextBlock(text="Data", bbox=[60, 0, 120, 20], confidence=0.8)
        ]
        
        result = OCRResult(text_blocks=blocks)
        assert abs(result.average_confidence - 0.85) < 0.01
        assert result.total_characters == 8
    
    def test_filter_by_confidence(self):
        """测试按置信度过滤"""
        blocks = [
            OCRTextBlock(text="High", bbox=[0, 0, 50, 20], confidence=0.9),
            OCRTextBlock(text="Low", bbox=[60, 0, 120, 20], confidence=0.5)
        ]
        
        result = OCRResult(text_blocks=blocks)
        filtered = result.filter_by_confidence(0.7)
        
        assert len(filtered) == 1
        assert filtered[0].text == "High"
    
    def test_sort_blocks(self):
        """测试排序"""
        blocks = [
            OCRTextBlock(text="C", bbox=[0, 40, 50, 60], confidence=0.7),
            OCRTextBlock(text="A", bbox=[0, 0, 50, 20], confidence=0.9),
            OCRTextBlock(text="B", bbox=[0, 20, 50, 40], confidence=0.8)
        ]
        
        result = OCRResult(text_blocks=blocks)
        
        # 按位置排序
        sorted_pos = result.sort_blocks(key='position')
        assert [b.text for b in sorted_pos.text_blocks] == ['A', 'B', 'C']
        
        # 按置信度排序
        sorted_conf = result.sort_blocks(key='confidence', reverse=True)
        assert [b.text for b in sorted_conf.text_blocks] == ['A', 'B', 'C']
    
    def test_result_to_dict(self):
        """测试转换为字典"""
        blocks = [OCRTextBlock(text="Test", bbox=[0, 0, 50, 20])]
        result = OCRResult(text_blocks=blocks, model_name="test-model")
        
        data = result.to_dict()
        assert 'text_blocks' in data
        assert 'full_text' in data
        assert 'num_blocks' in data
        assert data['model_name'] == "test-model"
    
    def test_result_from_text(self):
        """测试从纯文本创建"""
        result = OCRResult.from_text("Simple text")
        assert result.raw_text == "Simple text"
        assert result.full_text == "Simple text"
        assert len(result.text_blocks) == 0


class TestTokenDecoder:
    """测试 TokenDecoder"""
    
    def test_decoder_init(self):
        """测试解码器初始化"""
        mock_tokenizer = Mock()
        decoder = TokenDecoder(mock_tokenizer)
        
        assert decoder.tokenizer == mock_tokenizer
        assert decoder.skip_special_tokens == True
    
    def test_decode_single(self):
        """测试单个序列解码"""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "  Hello World  "
        
        decoder = TokenDecoder(mock_tokenizer)
        text = decoder.decode([1, 2, 3])
        
        assert text == "Hello World"
        mock_tokenizer.decode.assert_called_once()
    
    def test_batch_decode(self):
        """测试批量解码"""
        mock_tokenizer = Mock()
        mock_tokenizer.batch_decode.return_value = [
            "  Text 1  ",
            "  Text 2  "
        ]
        
        decoder = TokenDecoder(mock_tokenizer)
        texts = decoder.batch_decode([[1, 2], [3, 4]])
        
        assert texts == ["Text 1", "Text 2"]
        mock_tokenizer.batch_decode.assert_called_once()
    
    def test_decode_with_metadata(self):
        """测试带元数据的解码"""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Test"
        
        decoder = TokenDecoder(mock_tokenizer)
        result = decoder.decode_with_metadata([1, 2, 3])
        
        assert result['text'] == "Test"
        assert result['num_tokens'] == 3
        assert result['num_characters'] == 4
    
    def test_validate_output(self):
        """测试输出验证"""
        decoder = TokenDecoder(Mock())
        
        # 有效文本
        assert decoder.validate_output("Normal text") == True
        
        # 空文本
        assert decoder.validate_output("") == False
        assert decoder.validate_output("   ") == False
        
        # 过多特殊字符
        assert decoder.validate_output("!@#$%^&*()") == False


class TestResultParser:
    """测试 ResultParser"""
    
    def test_parser_init(self):
        """测试解析器初始化"""
        parser = ResultParser()
        assert parser.strict_mode == False
        
        parser_strict = ResultParser(strict_mode=True)
        assert parser_strict.strict_mode == True
    
    def test_parse_text_format(self):
        """测试解析纯文本"""
        parser = ResultParser()
        result = parser.parse("  Simple text  ", output_format="text")
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "Simple text"
        assert len(result.text_blocks) == 0
    
    def test_parse_json_standard(self):
        """测试解析标准 JSON 格式"""
        json_data = {
            "blocks": [
                {
                    "text": "Hello",
                    "bbox": [10, 20, 100, 50],
                    "confidence": 0.9
                },
                {
                    "text": "World",
                    "bbox": [110, 20, 200, 50],
                    "confidence": 0.8
                }
            ]
        }
        json_text = json.dumps(json_data)
        
        parser = ResultParser()
        result = parser.parse(json_text, output_format="json")
        
        assert len(result.text_blocks) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"
    
    def test_parse_json_code_block(self):
        """测试解析 JSON 代码块"""
        json_text = '''
        ```json
        {
            "blocks": [
                {"text": "Test", "bbox": [0, 0, 50, 20]}
            ]
        }
        ```
        '''
        
        parser = ResultParser()
        result = parser.parse(json_text, output_format="json")
        
        assert len(result.text_blocks) == 1
        assert result[0].text == "Test"
    
    def test_parse_json_array_format(self):
        """测试解析数组格式 JSON"""
        json_text = json.dumps([
            {"text": "A", "bbox": [0, 0, 50, 20]},
            {"text": "B", "bbox": [60, 0, 120, 20]}
        ])
        
        parser = ResultParser()
        result = parser.parse(json_text, output_format="json")
        
        assert len(result.text_blocks) == 2
    
    def test_parse_json_with_confidence_filter(self):
        """测试带置信度过滤的 JSON 解析"""
        json_data = {
            "blocks": [
                {"text": "High", "bbox": [0, 0, 50, 20], "confidence": 0.9},
                {"text": "Low", "bbox": [60, 0, 120, 20], "confidence": 0.3}
            ]
        }
        json_text = json.dumps(json_data)
        
        parser = ResultParser()
        result = parser.parse(json_text, output_format="json", confidence_threshold=0.5)
        
        assert len(result.text_blocks) == 1
        assert result[0].text == "High"
    
    def test_parse_json_fallback(self):
        """测试 JSON 解析失败降级"""
        parser = ResultParser(strict_mode=False)
        result = parser.parse("Not a JSON", output_format="json")
        
        assert isinstance(result, OCRResult)
        assert result.full_text == "Not a JSON"
    
    def test_parse_json_strict_mode(self):
        """测试严格模式"""
        parser = ResultParser(strict_mode=True)
        
        with pytest.raises(ValueError):
            parser.parse("Not a JSON", output_format="json")
    
    def test_parse_markdown(self):
        """测试解析 Markdown"""
        markdown_text = """
# Title
Paragraph 1
## Subtitle
Paragraph 2
        """
        
        parser = ResultParser()
        result = parser.parse(markdown_text, output_format="markdown")
        
        assert len(result.text_blocks) > 0
    
    def test_validate_result(self):
        """测试结果验证"""
        parser = ResultParser()
        
        # 有效结果
        valid_result = OCRResult.from_text("Valid text")
        assert parser.validate_result(valid_result) == True
        
        # 空结果
        empty_result = OCRResult.from_text("")
        assert parser.validate_result(empty_result) == False


class TestOutputFormatter:
    """测试 OutputFormatter"""
    
    def test_formatter_init(self):
        """测试格式化器初始化"""
        formatter = OutputFormatter()
        assert formatter.min_confidence == 0.0
        assert formatter.enable_deduplication == True
    
    def test_format_basic(self):
        """测试基础格式化"""
        blocks = [
            OCRTextBlock(text="Test", bbox=[0, 0, 50, 20], confidence=0.9)
        ]
        result = OCRResult(text_blocks=blocks)
        
        formatter = OutputFormatter()
        formatted = formatter.format(result)
        
        assert isinstance(formatted, OCRResult)
        assert len(formatted.text_blocks) == 1
    
    def test_coordinate_mapping(self):
        """测试坐标映射"""
        blocks = [
            OCRTextBlock(text="Test", bbox=[110, 120, 210, 170])
        ]
        result = OCRResult(text_blocks=blocks)
        
        transform_info = {
            'padding': (10, 20, 10, 20),  # left, top, right, bottom
            'scale': (2.0, 2.0),
            'original_size': (500, 500)
        }
        
        formatter = OutputFormatter()
        mapped = formatter.apply_coordinate_mapping(result, transform_info)
        
        # 还原后的坐标应该是 (100-10)/2=50, (120-20)/2=50
        assert abs(mapped[0].x1 - 50) < 1
        assert abs(mapped[0].y1 - 50) < 1
    
    def test_confidence_filtering(self):
        """测试置信度过滤"""
        blocks = [
            OCRTextBlock(text="High", bbox=[0, 0, 50, 20], confidence=0.9),
            OCRTextBlock(text="Low", bbox=[60, 0, 120, 20], confidence=0.3)
        ]
        result = OCRResult(text_blocks=blocks)
        
        formatter = OutputFormatter(min_confidence=0.5)
        filtered = formatter.format(result)
        
        assert len(filtered.text_blocks) == 1
        assert filtered[0].text == "High"
    
    def test_deduplication(self):
        """测试去重"""
        # 创建两个重叠的文本块
        blocks = [
            OCRTextBlock(text="Block1", bbox=[0, 0, 100, 50], confidence=0.9),
            OCRTextBlock(text="Block2", bbox=[10, 10, 110, 60], confidence=0.8)
        ]
        result = OCRResult(text_blocks=blocks)
        
        formatter = OutputFormatter(enable_deduplication=True, iou_threshold=0.3)
        deduped = formatter.deduplicate(result)
        
        # 应该只保留一个（置信度高的）
        assert len(deduped.text_blocks) == 1
        assert deduped[0].text == "Block1"
    
    def test_compute_iou(self):
        """测试 IoU 计算"""
        formatter = OutputFormatter()
        
        block1 = OCRTextBlock(text="A", bbox=[0, 0, 100, 100])
        block2 = OCRTextBlock(text="B", bbox=[50, 50, 150, 150])
        
        iou = formatter._compute_iou(block1, block2)
        
        # 交集: 50*50=2500, 并集: 10000+10000-2500=17500
        expected_iou = 2500 / 17500
        assert abs(iou - expected_iou) < 0.01
    
    def test_filter_small_blocks(self):
        """测试过滤小文本块"""
        blocks = [
            OCRTextBlock(text="Large", bbox=[0, 0, 100, 50]),
            OCRTextBlock(text="Small", bbox=[110, 0, 113, 2])
        ]
        result = OCRResult(text_blocks=blocks)
        
        formatter = OutputFormatter()
        filtered = formatter.filter_small_blocks(result, min_width=10, min_height=10)
        
        assert len(filtered.text_blocks) == 1
        assert filtered[0].text == "Large"
    
    def test_merge_adjacent_blocks(self):
        """测试合并相邻文本块"""
        blocks = [
            OCRTextBlock(text="Hello", bbox=[0, 0, 50, 20]),
            OCRTextBlock(text="World", bbox=[55, 0, 120, 20])
        ]
        result = OCRResult(text_blocks=blocks)
        
        formatter = OutputFormatter()
        merged = formatter.merge_adjacent_blocks(result, max_distance=10)
        
        # 应该合并为一个块
        assert len(merged.text_blocks) == 1
        assert "Hello" in merged[0].text
        assert "World" in merged[0].text
    
    def test_sort_blocks(self):
        """测试排序"""
        blocks = [
            OCRTextBlock(text="C", bbox=[0, 40, 50, 60]),
            OCRTextBlock(text="A", bbox=[0, 0, 50, 20]),
            OCRTextBlock(text="B", bbox=[0, 20, 50, 40])
        ]
        result = OCRResult(text_blocks=blocks)
        
        formatter = OutputFormatter()
        sorted_result = formatter.format(result, sort_by='position')
        
        texts = [b.text for b in sorted_result.text_blocks]
        assert texts == ['A', 'B', 'C']


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整的后处理流程"""
        # 1. 解码 (mock)
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = json.dumps({
            "blocks": [
                {"text": "Hello", "bbox": [110, 120, 210, 170], "confidence": 0.95},
                {"text": "World", "bbox": [220, 120, 320, 170], "confidence": 0.90}
            ]
        })
        
        decoder = TokenDecoder(mock_tokenizer)
        text = decoder.decode([1, 2, 3])
        
        # 2. 解析
        parser = ResultParser()
        result = parser.parse(text, output_format="json")
        
        assert len(result.text_blocks) == 2
        
        # 3. 格式化
        transform_info = {
            'padding': (10, 20, 10, 20),
            'scale': (2.0, 2.0)
        }
        
        formatter = OutputFormatter(min_confidence=0.85)
        final_result = formatter.format(result, transform_info=transform_info)
        
        # 验证结果
        assert len(final_result.text_blocks) == 2
        # 坐标应该被映射
        assert final_result[0].x1 < 110  # 应该小于原始值
    
    def test_error_handling(self):
        """测试错误处理"""
        parser = ResultParser(strict_mode=False)
        
        # 空文本
        result1 = parser.parse("", output_format="json")
        assert isinstance(result1, OCRResult)
        
        # 无效 JSON
        result2 = parser.parse("invalid json {{{", output_format="json")
        assert isinstance(result2, OCRResult)
    
    def test_edge_cases(self):
        """测试边界情况"""
        formatter = OutputFormatter()
        
        # 空结果
        empty_result = OCRResult()
        formatted = formatter.format(empty_result)
        assert len(formatted.text_blocks) == 0
        
        # 单个块
        single_block = OCRResult(text_blocks=[
            OCRTextBlock(text="Single", bbox=[0, 0, 50, 20])
        ])
        formatted_single = formatter.format(single_block)
        assert len(formatted_single.text_blocks) == 1
