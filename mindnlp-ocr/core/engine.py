"""
VLM-OCR主引擎
协调各个组件完成端到端的OCR流程
支持自适应批处理优化
"""

import time
from typing import List, Dict, Any
from api.schemas.request import OCRRequest, OCRBatchRequest, OCRURLRequest
from api.schemas.response import OCRResponse
from .processor.image import ImageProcessor
from .processor.prompt import PromptBuilder
from .processor.batch import AdaptiveBatchCollator
from .parser.decoder import TokenDecoder
from .parser.result import ResultParser
from .parser.formatter import OutputFormatter
from .validator.input import InputValidator
from models.loader import ModelLoader
from utils.logger import get_logger
from utils.image_utils import download_image_from_url


logger = get_logger(__name__)


class VLMOCREngine:
    """VLM-OCR主引擎"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        enable_adaptive_batching: bool = True,
        max_batch_size: int = 8
    ):
        """
        初始化OCR引擎
        
        Args:
            model_name: 模型名称
            device: 运行设备
            enable_adaptive_batching: 是否启用自适应批处理
            max_batch_size: 最大批大小
        """
        logger.info(f"Initializing VLM-OCR Engine with model: {model_name}")
        
        # 保存配置
        self.model_name = model_name
        self.device = device
        
        # 加载模型
        self.model_loader = ModelLoader(model_name, device)
        self.model = self.model_loader.load_model()
        self.tokenizer = self.model_loader.load_tokenizer()
        
        # 初始化组件
        self.image_processor = ImageProcessor()
        self.prompt_builder = PromptBuilder()
        self.token_decoder = TokenDecoder(self.tokenizer)
        self.result_parser = ResultParser()
        self.output_formatter = OutputFormatter()
        self.input_validator = InputValidator()
        
        # 初始化自适应批处理整理器
        self.enable_adaptive_batching = enable_adaptive_batching
        if enable_adaptive_batching:
            self.batch_collator = AdaptiveBatchCollator(
                max_batch_size=max_batch_size,
                size_threshold=0.2,
                enable_dynamic_batching=True
            )
            logger.info("Adaptive batching enabled")
        else:
            self.batch_collator = None
        
        logger.info("VLM-OCR Engine initialized successfully")
    
    def predict(self, request: OCRRequest) -> OCRResponse:
        """
        单张图像OCR预测
        
        Args:
            request: OCR请求
            
        Returns:
            OCRResponse: OCR识别结果
        """
        start_time = time.time()
        
        try:
            # 1. 输入验证
            self.input_validator.validate_image(request.image)
            self.input_validator.validate_params(
                output_format=request.output_format,
                language=request.language,
                task_type=request.task_type
            )
            
            # 2. 图像预处理
            logger.info("Processing image...")
            processed_image = self.image_processor.process(request.image)
            
            # 3. 构建Prompt
            logger.info(f"Building prompt for task: {request.task_type}")
            prompt = self.prompt_builder.build(
                task_type=request.task_type,
                output_format=request.output_format,
                language=request.language,
                custom_prompt=request.custom_prompt
            )
            
            # 4. 准备模型输入
            inputs = self._prepare_inputs(processed_image, prompt)
            
            # 5. 模型推理
            logger.info("Running model inference...")
            outputs = self.model.generate(**inputs)
            
            # 6. Token解码
            decoded_text = self.token_decoder.decode(outputs[0])
            
            # 7. 结果解析
            parsed_result = self.result_parser.parse(
                decoded_text,
                output_format=request.output_format,
                confidence_threshold=request.confidence_threshold
            )
            
            # 8. 格式化输出
            formatted_result = self.output_formatter.format(
                parsed_result,
                output_format=request.output_format
            )
            
            # 构建响应
            processing_time = time.time() - start_time
            return OCRResponse(
                success=True,
                texts=formatted_result.get('texts', []),
                boxes=formatted_result.get('boxes', []),
                confidences=formatted_result.get('confidences', []),
                raw_output=decoded_text,
                inference_time=processing_time,
                model_name=self.model_name,
                metadata={
                    'format': request.output_format,
                    'language': request.language,
                    'task_type': request.task_type
                }
            )
            
        except Exception as e:
            logger.error(f"OCR prediction failed: {str(e)}")
            processing_time = time.time() - start_time
            return OCRResponse(
                success=False,
                texts=[],
                boxes=[],
                confidences=[],
                raw_output="",
                inference_time=processing_time,
                model_name=self.model_name,
                metadata={
                    'format': request.output_format,
                    'language': request.language
                },
                error=str(e)
            )
    
    def predict_batch(self, request: OCRBatchRequest) -> List[OCRResponse]:
        """
        批量图像OCR预测（使用自适应批处理）
        
        Args:
            request: 批量OCR请求
            
        Returns:
            List[OCRResponse]: OCR识别结果列表
        """
        logger.info(f"Processing batch of {len(request.images)} images...")
        
        if self.enable_adaptive_batching and len(request.images) > 1:
            return self._predict_batch_adaptive(request)
        else:
            return self._predict_batch_sequential(request)
    
    def _predict_batch_sequential(self, request: OCRBatchRequest) -> List[OCRResponse]:
        """顺序批处理（逐个处理）"""
        results = []
        
        for idx, image in enumerate(request.images):
            logger.info(f"Processing image {idx + 1}/{len(request.images)}")
            single_request = OCRRequest(
                image=image,
                output_format=request.output_format,
                language=request.language,
                task_type=request.task_type,
                confidence_threshold=request.confidence_threshold,
                custom_prompt=request.custom_prompt
            )
            result = self.predict(single_request)
            results.append(result)
        
        return results
    
    def _predict_batch_adaptive(self, request: OCRBatchRequest) -> List[OCRResponse]:
        """自适应批处理（按尺寸分组并行处理）"""
        logger.info("Using adaptive batching...")
        
        try:
            # 1. 预处理所有图像
            processed_images = []
            for img in request.images:
                self.input_validator.validate_image(img)
                processed_img = self.image_processor.process(img)
                processed_images.append(processed_img)
            
            # 2. 使用自适应批处理整理器分组
            batches = self.batch_collator.collate_adaptive(
                images=processed_images,
                auto_group=True
            )
            
            logger.info(f"Split into {len(batches)} optimized batches")
            
            # 3. 处理每个批次
            all_results = []
            for batch_idx, (batch_tensor, batch_info) in enumerate(batches):
                batch_size = batch_info['batch_size']
                logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}, size={batch_size}")
                
                # 构建prompts
                prompts = [
                    self.prompt_builder.build_prompt(
                        task_type=request.task_type,
                        output_format=request.output_format,
                        language=request.language,
                        custom_prompt=request.custom_prompt
                    )
                    for _ in range(batch_size)
                ]
                
                # 批量推理
                batch_results = self._batch_inference(
                    batch_tensor,
                    prompts,
                    request
                )
                
                all_results.extend(batch_results)
            
            logger.info(f"Adaptive batch processing completed: {len(all_results)} results")
            return all_results
            
        except Exception as e:
            logger.error(f"Adaptive batch processing failed, falling back to sequential: {e}")
            return self._predict_batch_sequential(request)
    
    def _batch_inference(
        self,
        batch_tensor: Any,
        prompts: List[str],
        request: OCRBatchRequest
    ) -> List[OCRResponse]:
        """
        批量推理
        
        Args:
            batch_tensor: 批量图像Tensor
            prompts: Prompt列表
            request: 批量请求
            
        Returns:
            List[OCRResponse]: 批量结果
        """
        batch_start = time.time()
        results = []
        
        try:
            # 准备批量输入
            # 注意：这里简化处理，实际需要根据具体模型调整
            for i, prompt in enumerate(prompts):
                # 提取单张图像
                single_image = batch_tensor[i:i+1]
                
                # 准备输入
                inputs = self._prepare_inputs(single_image, prompt)
                
                # 推理
                outputs = self.model.generate(**inputs)
                
                # 解码
                decoded_text = self.token_decoder.decode(outputs[0])
                
                # 解析
                parsed_result = self.result_parser.parse(
                    decoded_text,
                    output_format=request.output_format,
                    confidence_threshold=request.confidence_threshold
                )
                
                # 格式化
                formatted_result = self.output_formatter.format(
                    parsed_result,
                    output_format=request.output_format
                )
                
                # 构建响应
                inference_time = time.time() - batch_start
                result = OCRResponse(
                    success=True,
                    text=formatted_result.get('text', ''),
                    blocks=formatted_result.get('blocks'),
                    format=request.output_format,
                    language=request.language,
                    processing_time=inference_time
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # 返回失败响应
            for _ in range(len(prompts)):
                results.append(OCRResponse(
                    success=False,
                    text="",
                    format=request.output_format,
                    language=request.language,
                    processing_time=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    def process_batch_data(self, batch_data: List[Dict[str, Any]]) -> List[OCRResponse]:
        """
        处理批量数据（用于AsyncBatchQueue）
        
        Args:
            batch_data: 批量请求数据列表
            
        Returns:
            List[OCRResponse]: 批量结果
        """
        logger.info(f"Processing batch data: {len(batch_data)} items")
        
        try:
            # 构建OCR请求
            images = [data['image'] for data in batch_data]
            
            # 使用第一个请求的参数（假设批次中参数相同）
            first_data = batch_data[0]
            batch_request = OCRBatchRequest(
                images=images,
                output_format=first_data.get('output_format', 'text'),
                language=first_data.get('language', 'auto'),
                task_type=first_data.get('task_type', 'general'),
                confidence_threshold=first_data.get('confidence_threshold', 0.0),
                custom_prompt=first_data.get('custom_prompt')
            )
            
            # 执行批处理
            results = self.predict_batch(batch_request)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch data processing failed: {e}")
            # 返回失败响应
            return [
                OCRResponse(
                    success=False,
                    text="",
                    format="text",
                    language="auto",
                    processing_time=0.0,
                    error_message=str(e)
                )
                for _ in batch_data
            ]
    
    def predict_from_url(self, request: OCRURLRequest) -> OCRResponse:
        """
        从URL预测OCR
        
        Args:
            request: URL OCR请求
            
        Returns:
            OCRResponse: OCR识别结果
        """
        logger.info(f"Downloading image from URL: {request.image_url}")
        image_bytes = download_image_from_url(str(request.image_url))
        
        single_request = OCRRequest(
            image=image_bytes,
            output_format=request.output_format,
            language=request.language,
            task_type=request.task_type,
            confidence_threshold=request.confidence_threshold,
            custom_prompt=request.custom_prompt
        )
        
        return self.predict(single_request)
    
    def _prepare_inputs(self, image, prompt):
        """
        准备模型输入
        
        Args:
            image: 处理后的图像
            prompt: 构建的Prompt
            
        Returns:
            dict: 模型输入字典
        """
        # 这里使用transformers的标准接口
        # 具体实现依赖于所使用的VLM模型
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        
        # 添加图像输入 (具体格式取决于模型)
        inputs['pixel_values'] = image
        
        return inputs
