"""
OCR预测路由
支持同步和异步批处理
"""

import time
import uuid
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from ..schemas.request import OCRRequest, OCRBatchRequest, OCRURLRequest
from ..schemas.response import OCRResponse, BatchOCRResponse
from utils.logger import get_logger
from config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

# 全局批处理队列（将在app.py中初始化）
_batch_queue = None


def set_batch_queue(queue):
    """设置批处理队列"""
    global _batch_queue
    _batch_queue = queue


def get_batch_queue():
    """获取批处理队列"""
    if _batch_queue is None:
        raise RuntimeError("Batch queue not initialized")
    return _batch_queue


def get_engine():
    """获取OCR引擎实例（延迟导入避免循环依赖）"""
    from ..app import get_engine as _get_engine
    return _get_engine()


@router.post("/predict", response_model=OCRResponse)
async def predict_image(
    file: UploadFile = File(...),
    output_format: str = Form("text"),
    language: str = Form("auto"),
    task_type: str = Form("general"),
    confidence_threshold: float = Form(0.0),
    model: str = Form(None)  # 新增：模型选择参数
):
    """
    单张图像OCR预测
    
    Args:
        file: 上传的图像文件
        output_format: 输出格式 (text/json/markdown)
        language: 语言设置 (auto/zh/en/ja/ko)
        task_type: 任务类型 (general/document/table/formula)
        confidence_threshold: 置信度阈值
        model: 模型类型 (qwen2-vl/internvl/llava，可选)
    
    Returns:
        OCRResponse: OCR识别结果
    """
    start_time = time.time()
    
    try:
        # 验证模型类型
        if model and model not in settings.available_models:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的模型: {model}。可用模型: {list(settings.available_models.keys())}"
            )
        
        # 使用指定模型或默认模型
        model_type = model or settings.default_model
        
        # 验证文件类型
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Only image files are allowed."
            )
        
        # 获取引擎
        engine = get_engine(model_type)  # 传入模型类型
        
        # 读取图像数据
        image_bytes = await file.read()
        
        # 构建请求
        request = OCRRequest(
            image=image_bytes,
            output_format=output_format,
            language=language,
            task_type=task_type,
            confidence_threshold=confidence_threshold
        )
        
        # 执行OCR (这里暂时返回模拟数据，等待engine.predict实现)
        # result = engine.predict(request)
        
        # 模拟响应
        inference_time = time.time() - start_time
        result = OCRResponse(
            success=True,
            texts=["识别的文本内容"],
            boxes=[[10, 20, 200, 30]],
            confidences=[0.95],
            raw_output="识别的文本内容",
            inference_time=inference_time,
            model_name=settings.default_model,
            metadata={
                "language": language,
                "format": output_format,
                "task_type": task_type
            }
        )
        
        return result
        
    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR prediction failed: {str(e)}")


@router.post("/predict_batch", response_model=BatchOCRResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    output_format: str = Form("text"),
    language: str = Form("auto"),
    task_type: str = Form("general"),
    confidence_threshold: float = Form(0.0)
):
    """
    批量图像OCR预测
    
    Args:
        files: 上传的图像文件列表
        output_format: 输出格式 (text/json/markdown)
        language: 语言设置 (auto/zh/en/ja/ko)
        task_type: 任务类型 (general/document/table/formula)
        confidence_threshold: 置信度阈值
    
    Returns:
        BatchOCRResponse: OCR识别结果列表
    """
    start_time = time.time()
    
    try:
        # 获取引擎
        engine = get_engine()
        
        # 处理每个图像
        results = []
        for file in files:
            image_bytes = await file.read()
            
            # 执行单张OCR
            request = OCRRequest(
                image=image_bytes,
                output_format=output_format,
                language=language,
                task_type=task_type,
                confidence_threshold=confidence_threshold
            )
            
            # 模拟单张处理
            single_result = OCRResponse(
                success=True,
                texts=["文本内容"],
                boxes=[[10, 20, 200, 30]],
                confidences=[0.95],
                raw_output="文本内容",
                inference_time=0.5,
                model_name=settings.default_model,
                metadata={"language": language}
            )
            results.append(single_result)
        
        total_time = time.time() - start_time
        
        return BatchOCRResponse(
            success=True,
            results=results,
            total_images=len(files),
            total_time=total_time,
            model_name=settings.default_model
        )
        
    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Batch OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch OCR prediction failed: {str(e)}")


@router.post("/predict_url", response_model=OCRResponse)
async def predict_from_url(request: OCRURLRequest):
    """
    从URL预测OCR
    
    Args:
        request: 包含图像URL的请求
    
    Returns:
        OCRResponse: OCR识别结果
    """
    start_time = time.time()
    
    try:
        # 获取引擎
        engine = get_engine()
        
        # 下载图像
        from utils.image_utils import download_image_from_url
        image_bytes = download_image_from_url(str(request.image_url))
        
        # 执行OCR
        ocr_request = OCRRequest(
            image=image_bytes,
            output_format=request.output_format,
            language=request.language,
            task_type=request.task_type,
            confidence_threshold=request.confidence_threshold
        )
        
        # 模拟响应
        inference_time = time.time() - start_time
        result = OCRResponse(
            success=True,
            texts=["URL图像识别的文本"],
            boxes=[[10, 20, 200, 30]],
            confidences=[0.95],
            raw_output="URL图像识别的文本",
            inference_time=inference_time,
            model_name=settings.default_model,
            metadata={"source": "url"}
        )
        
        return result
        
    except HTTPException:
        # 重新抛出HTTPException，不要捕获
        raise
    except RuntimeError as e:
        logger.error(f"Engine not ready: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"URL OCR prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL OCR prediction failed: {str(e)}")


@router.post("/predict_batch_async", response_model=BatchOCRResponse)
async def predict_batch_async(
    files: List[UploadFile] = File(...),
    output_format: str = Form("text"),
    language: str = Form("auto"),
    task_type: str = Form("general"),
    confidence_threshold: float = Form(0.0),
    timeout: Optional[float] = Form(30.0)
):
    """
    异步批量图像OCR预测（使用批处理队列）
    
    特性:
    - 自动批处理聚合
    - 智能等待策略
    - 更高的吞吐量
    
    Args:
        files: 上传的图像文件列表
        output_format: 输出格式 (text/json/markdown)
        language: 语言设置 (auto/zh/en/ja/ko)
        task_type: 任务类型 (general/document/table/formula)
        confidence_threshold: 置信度阈值
        timeout: 请求超时时间（秒）
    
    Returns:
        BatchOCRResponse: OCR识别结果列表
    """
    start_time = time.time()
    
    try:
        # 获取批处理队列
        batch_queue = get_batch_queue()
        
        # 读取所有图像
        images_data = []
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}"
                )
            
            image_bytes = await file.read()
            images_data.append({
                'image': image_bytes,
                'filename': file.filename
            })
        
        # 为每个图像创建请求并提交到队列
        import asyncio
        futures = []
        for idx, img_data in enumerate(images_data):
            request_id = f"{uuid.uuid4()}-{idx}"
            
            # 构建OCR请求数据
            request_data = {
                'image': img_data['image'],
                'output_format': output_format,
                'language': language,
                'task_type': task_type,
                'confidence_threshold': confidence_threshold,
                'filename': img_data['filename']
            }
            
            # 提交到队列
            future = batch_queue.add_request(
                request_id=request_id,
                data=request_data,
                timeout=timeout
            )
            futures.append(future)
        
        # 等待所有结果
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 处理结果
        ocr_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {idx} failed: {result}")
                # 创建失败响应
                ocr_results.append(OCRResponse(
                    success=False,
                    texts=[],
                    boxes=[],
                    confidences=[],
                    raw_output="",
                    inference_time=0.0,
                    model_name=settings.default_model,
                    metadata={"error": str(result)}
                ))
            else:
                ocr_results.append(result)
        
        total_time = time.time() - start_time
        
        return BatchOCRResponse(
            success=True,
            results=ocr_results,
            total_images=len(files),
            total_time=total_time,
            model_name=settings.default_model
        )
        
    except RuntimeError as e:
        logger.error(f"Batch queue not available: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Async batch OCR failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Async batch OCR failed: {str(e)}")


@router.get("/batch_queue/metrics")
async def get_batch_queue_metrics():
    """
    获取批处理队列指标
    
    Returns:
        dict: 队列指标信息
    """
    try:
        batch_queue = get_batch_queue()
        metrics = batch_queue.get_metrics()
        
        return {
            "success": True,
            "metrics": metrics
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """
    列出所有可用的模型
    
    Returns:
        dict: 模型列表和详细信息
    """
    try:
        models_info = []
        
        for model_type, model_path in settings.available_models.items():
            model_info = {
                "type": model_type,
                "model_id": model_path,
                "description": settings.model_descriptions.get(model_type, ""),
                "capabilities": settings.model_capabilities.get(model_type, []),
                "is_default": model_type == settings.default_model
            }
            models_info.append(model_info)
        
        return {
            "success": True,
            "default_model": settings.default_model,
            "total_models": len(models_info),
            "models": models_info,
            "multi_model_enabled": settings.enable_multi_model
        }
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_type}")
async def get_model_info(model_type: str):
    """
    获取特定模型的详细信息
    
    Args:
        model_type: 模型类型
        
    Returns:
        dict: 模型详细信息
    """
    try:
        if model_type not in settings.available_models:
            raise HTTPException(
                status_code=404,
                detail=f"模型不存在: {model_type}"
            )
        
        # 导入ModelFactory获取更多信息
        from models.loader import ModelFactory
        factory_info = ModelFactory.get_model_info(model_type)
        
        model_info = {
            "type": model_type,
            "model_id": settings.available_models[model_type],
            "description": settings.model_descriptions.get(model_type, ""),
            "capabilities": settings.model_capabilities.get(model_type, []),
            "is_default": model_type == settings.default_model,
            **factory_info
        }
        
        return {
            "success": True,
            "model": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/compare")
async def compare_models(
    file: UploadFile = File(...),
    prompt: str = Form("识别图中的文字"),
    models: List[str] = Form(None)
):
    """
    使用多个模型对比识别同一图像
    
    Args:
        file: 上传的图像文件
        prompt: OCR提示词
        models: 要对比的模型列表（不提供则使用所有可用模型）
        
    Returns:
        dict: 各模型的识别结果对比
    """
    start_time = time.time()
    
    try:
        # 验证文件类型
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}"
            )
        
        # 确定要对比的模型
        if models is None or len(models) == 0:
            models = list(settings.available_models.keys())
        else:
            # 验证模型
            invalid_models = [m for m in models if m not in settings.available_models]
            if invalid_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的模型: {invalid_models}"
                )
        
        # 读取图像
        image_bytes = await file.read()
        
        # 对每个模型进行推理
        results = {}
        for model_type in models:
            model_start = time.time()
            
            try:
                # 获取该模型的引擎
                engine = get_engine(model_type)
                
                # 执行OCR（这里是模拟）
                result = f"[{model_type}] 识别结果（模拟）"
                
                model_time = time.time() - model_start
                
                results[model_type] = {
                    "success": True,
                    "result": result,
                    "inference_time": model_time,
                    "model_path": settings.available_models[model_type]
                }
                
            except Exception as e:
                logger.error(f"Model {model_type} failed: {e}")
                results[model_type] = {
                    "success": False,
                    "error": str(e),
                    "model_path": settings.available_models[model_type]
                }
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "prompt": prompt,
            "models_tested": len(models),
            "results": results,
            "total_time": total_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

