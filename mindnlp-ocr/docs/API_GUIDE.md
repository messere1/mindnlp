# API 使用指南

## 目录

- [快速开始](#快速开始)
- [认证](#认证)
- [端点详情](#端点详情)
- [请求格式](#请求格式)
- [响应格式](#响应格式)
- [错误处理](#错误处理)
- [最佳实践](#最佳实践)

## 快速开始

### 启动服务

```bash
# 开发模式
python -m api.app

# 生产模式
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4

# Docker 部署
docker build -t vlm-ocr .
docker run -p 8000:8000 vlm-ocr
```

### 访问 API 文档

启动服务后，访问以下地址：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 认证

当前版本不需要认证。生产环境建议添加 API Key 或 OAuth2。

## 端点详情

### 1. 健康检查

#### GET /api/v1/health

检查服务是否运行。

**请求示例**:
```bash
curl http://localhost:8000/api/v1/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

#### GET /api/v1/health/ready

检查服务是否准备好接受请求（模型已加载）。

**请求示例**:
```bash
curl http://localhost:8000/api/v1/health/ready
```

**响应示例**:
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2-VL-2B-Instruct",
  "device": "cuda"
}
```

### 2. 单图像 OCR

#### POST /api/v1/ocr/predict

对单张图像进行 OCR 识别。

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| file | File | 是 | - | 图像文件 |
| output_format | string | 否 | json | 输出格式：json/text/markdown |
| language | string | 否 | zh | 语言：zh/en/ja/ko/multi |
| task_type | string | 否 | general | 任务类型：general/document/table/formula |
| confidence_threshold | float | 否 | 0.5 | 置信度阈值 (0.0-1.0) |
| custom_prompt | string | 否 | null | 自定义 Prompt |

**请求示例** (Python):
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/predict",
        files={"file": f},
        data={
            "output_format": "json",
            "language": "zh",
            "task_type": "general",
            "confidence_threshold": 0.8
        }
    )

result = response.json()
print(result)
```

**请求示例** (cURL):
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "output_format=json" \
  -F "language=zh" \
  -F "confidence_threshold=0.8"
```

**响应示例**:
```json
{
  "success": true,
  "texts": [
    "这是第一行文本",
    "这是第二行文本"
  ],
  "boxes": [
    [10, 20, 100, 40],
    [10, 50, 120, 70]
  ],
  "confidences": [0.95, 0.92],
  "inference_time": 0.523,
  "format": "json",
  "language": "zh",
  "task_type": "general"
}
```

### 3. 批量 OCR

#### POST /api/v1/ocr/predict_batch

对多张图像进行批量 OCR 识别。

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| files | List[File] | 是 | - | 图像文件列表 |
| output_format | string | 否 | json | 输出格式 |
| language | string | 否 | zh | 语言 |
| task_type | string | 否 | general | 任务类型 |
| confidence_threshold | float | 否 | 0.5 | 置信度阈值 |

**请求示例** (Python):
```python
import requests

files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]

response = requests.post(
    "http://localhost:8000/api/v1/ocr/predict_batch",
    files=files,
    data={
        "output_format": "json",
        "language": "zh",
        "confidence_threshold": 0.8
    }
)

results = response.json()
for i, result in enumerate(results["results"]):
    print(f"Image {i+1}:")
    print(f"  Texts: {result['texts']}")
    print(f"  Time: {result['inference_time']}s")
```

**请求示例** (cURL):
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/predict_batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "output_format=json" \
  -F "language=zh"
```

**响应示例**:
```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "texts": ["文本1"],
      "boxes": [[10, 20, 100, 40]],
      "confidences": [0.95],
      "inference_time": 0.5
    },
    {
      "success": true,
      "texts": ["文本2"],
      "boxes": [[15, 25, 110, 45]],
      "confidences": [0.93],
      "inference_time": 0.48
    }
  ],
  "total_time": 1.2,
  "count": 2
}
```

### 4. URL 图像 OCR

#### POST /api/v1/ocr/predict_url

对 URL 指定的图像进行 OCR 识别。

**请求参数**:

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| image_url | string | 是 | - | 图像 URL |
| output_format | string | 否 | json | 输出格式 |
| language | string | 否 | zh | 语言 |
| task_type | string | 否 | general | 任务类型 |
| confidence_threshold | float | 否 | 0.5 | 置信度阈值 |

**请求示例** (Python):
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/ocr/predict_url",
    json={
        "image_url": "https://example.com/image.jpg",
        "output_format": "json",
        "language": "en",
        "confidence_threshold": 0.8
    }
)

result = response.json()
print(result)
```

**请求示例** (cURL):
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/predict_url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "output_format": "json",
    "language": "en",
    "confidence_threshold": 0.8
  }'
```

**响应示例**:
```json
{
  "success": true,
  "texts": ["Sample text"],
  "boxes": [[20, 30, 150, 60]],
  "confidences": [0.94],
  "inference_time": 0.6,
  "format": "json",
  "language": "en",
  "task_type": "general"
}
```

## 请求格式

### 图像格式

支持以下图像格式：
- JPEG / JPG
- PNG
- BMP
- TIFF
- WebP

### 图像大小

- **最小尺寸**: 32x32 像素
- **最大尺寸**: 4096x4096 像素（自动缩放）
- **文件大小**: 最大 10MB

### 输出格式

#### 1. JSON 格式 (`output_format=json`)

结构化的 JSON 输出，包含文本、坐标、置信度：

```json
{
  "texts": ["文本1", "文本2"],
  "boxes": [[x1, y1, x2, y2], [x1, y1, x2, y2]],
  "confidences": [0.95, 0.92]
}
```

坐标说明：
- `x1, y1`: 左上角坐标
- `x2, y2`: 右下角坐标
- 坐标相对于原始图像尺寸

#### 2. Text 格式 (`output_format=text`)

纯文本输出，一行一个文本块：

```
文本1
文本2
文本3
```

#### 3. Markdown 格式 (`output_format=markdown`)

Markdown 格式输出，适合文档：

```markdown
# 标题
段落内容...

## 子标题
更多内容...
```

### 语言选项

- `zh`: 中文
- `en`: 英文
- `ja`: 日语
- `ko`: 韩语
- `multi`: 多语言混合（自动检测）

### 任务类型

- `general`: 通用 OCR（默认）
- `document`: 文档理解（保留结构）
- `table`: 表格识别（结构化输出）
- `formula`: 公式识别（LaTeX 格式）

## 响应格式

### 成功响应

```json
{
  "success": true,
  "texts": [...],
  "boxes": [...],
  "confidences": [...],
  "inference_time": 0.5,
  "format": "json",
  "language": "zh",
  "task_type": "general"
}
```

### 错误响应

```json
{
  "success": false,
  "error": "错误类型",
  "message": "详细错误信息",
  "status_code": 400
}
```

## 错误处理

### HTTP 状态码

| 状态码 | 说明 | 常见原因 |
|--------|------|----------|
| 200 | 成功 | 请求成功处理 |
| 400 | 请求错误 | 参数无效、图像格式错误 |
| 413 | 载荷过大 | 图像文件超过限制 |
| 422 | 验证错误 | 参数类型错误 |
| 500 | 服务器错误 | 内部处理错误 |
| 503 | 服务不可用 | 模型未加载 |

### 常见错误

#### 1. 图像格式错误

**错误信息**:
```json
{
  "error": "InvalidImageFormat",
  "message": "Unsupported image format. Supported: JPEG, PNG, BMP, TIFF, WebP"
}
```

**解决方法**:
- 检查图像格式
- 转换为支持的格式

#### 2. 图像过大

**错误信息**:
```json
{
  "error": "ImageTooLarge",
  "message": "Image size exceeds maximum: 10MB"
}
```

**解决方法**:
- 压缩图像
- 降低分辨率

#### 3. 参数无效

**错误信息**:
```json
{
  "error": "ValidationError",
  "message": "confidence_threshold must be between 0 and 1"
}
```

**解决方法**:
- 检查参数值
- 参考 API 文档

#### 4. 模型未加载

**错误信息**:
```json
{
  "error": "ModelNotReady",
  "message": "Model is still loading. Please wait..."
}
```

**解决方法**:
- 等待模型加载完成
- 检查 `/health/ready` 端点

## 最佳实践

### 1. 性能优化

#### 使用批量处理

```python
# ❌ 不推荐：多次单独请求
for image in images:
    response = requests.post(url, files={"file": image})

# ✅ 推荐：批量处理
files = [("files", img) for img in images]
response = requests.post(url + "/predict_batch", files=files)
```

#### 复用连接

```python
# 使用 Session 复用连接
session = requests.Session()
for image in images:
    response = session.post(url, files={"file": image})
```

### 2. 错误处理

```python
import requests
from requests.exceptions import RequestException

def ocr_with_retry(image_path, max_retries=3):
    session = requests.Session()
    
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                response = session.post(
                    "http://localhost:8000/api/v1/ocr/predict",
                    files={"file": f},
                    timeout=30
                )
            
            response.raise_for_status()
            return response.json()
            
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)  # 指数退避
```

### 3. 参数选择

#### 置信度阈值

```python
# 高精度场景（金融、法律）
confidence_threshold = 0.9

# 平衡场景（通用 OCR）
confidence_threshold = 0.7

# 低置信度场景（模糊图像）
confidence_threshold = 0.5
```

#### 输出格式

```python
# 需要坐标信息 → JSON
output_format = "json"

# 纯文本提取 → Text
output_format = "text"

# 文档理解 → Markdown
output_format = "markdown"
```

### 4. 异步处理

```python
import asyncio
import aiohttp

async def ocr_async(session, image_path):
    async with aiohttp.ClientSession() as session:
        with open(image_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename="image.jpg")
            
            async with session.post(url, data=data) as response:
                return await response.json()

async def batch_ocr_async(image_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [ocr_async(session, path) for path in image_paths]
        return await asyncio.gather(*tasks)

# 使用
results = asyncio.run(batch_ocr_async(image_paths))
```

### 5. 监控和日志

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ocr_with_logging(image_path):
    start = time.time()
    
    try:
        with open(image_path, "rb") as f:
            response = requests.post(url, files={"file": f})
            result = response.json()
        
        elapsed = time.time() - start
        logger.info(f"OCR success: {image_path}, time: {elapsed:.2f}s")
        return result
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"OCR failed: {image_path}, time: {elapsed:.2f}s, error: {e}")
        raise
```

## Python 客户端库

### 简单封装

```python
from typing import List, Optional
import requests

class VLMOCRClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def predict(
        self,
        image_path: str,
        output_format: str = "json",
        language: str = "zh",
        confidence_threshold: float = 0.5
    ) -> dict:
        """单图像 OCR"""
        with open(image_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/api/v1/ocr/predict",
                files={"file": f},
                data={
                    "output_format": output_format,
                    "language": language,
                    "confidence_threshold": confidence_threshold
                }
            )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(
        self,
        image_paths: List[str],
        output_format: str = "json",
        language: str = "zh",
        confidence_threshold: float = 0.5
    ) -> dict:
        """批量 OCR"""
        files = [("files", open(path, "rb")) for path in image_paths]
        
        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/predict_batch",
            files=files,
            data={
                "output_format": output_format,
                "language": language,
                "confidence_threshold": confidence_threshold
            }
        )
        response.raise_for_status()
        return response.json()
    
    def predict_url(
        self,
        image_url: str,
        output_format: str = "json",
        language: str = "zh",
        confidence_threshold: float = 0.5
    ) -> dict:
        """URL 图像 OCR"""
        response = self.session.post(
            f"{self.base_url}/api/v1/ocr/predict_url",
            json={
                "image_url": image_url,
                "output_format": output_format,
                "language": language,
                "confidence_threshold": confidence_threshold
            }
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> dict:
        """健康检查"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def ready(self) -> dict:
        """就绪检查"""
        response = self.session.get(f"{self.base_url}/api/v1/health/ready")
        response.raise_for_status()
        return response.json()

# 使用示例
client = VLMOCRClient()

# 检查服务状态
print(client.health())
print(client.ready())

# 单图像 OCR
result = client.predict("image.jpg", language="zh")
print(result)

# 批量 OCR
results = client.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
print(results)

# URL OCR
result = client.predict_url("https://example.com/image.jpg")
print(result)
```

## 总结

本 API 提供了完整的 OCR 功能，支持：
- ✅ 多种输入方式（文件、批量、URL）
- ✅ 多种输出格式（JSON、Text、Markdown）
- ✅ 多语言支持
- ✅ 灵活的参数配置
- ✅ 完善的错误处理

更多信息请参考：
- [README](../README_INTEGRATION.md)
- [示例代码](../examples/)
- [API 在线文档](http://localhost:8000/docs)
