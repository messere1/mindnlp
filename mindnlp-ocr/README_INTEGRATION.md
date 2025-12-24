# MindNLP VLM-OCR Engine

基于视觉语言模型（VLM）的 OCR 引擎，支持多语言文本识别、文档理解、表格识别等功能。

## 特性

- 🌐 **多语言支持**：中文、英文、日语、韩语及多语言混合
- 📄 **多种任务类型**：通用 OCR、文档理解、表格识别、公式识别
- 📝 **灵活输出格式**：JSON、Text、Markdown
- 🚀 **高性能**：支持批量处理、智能缓存
- 🔧 **易于集成**：RESTful API 和 Python SDK
- 🎯 **高精度**：基于最新的 VLM 模型（Qwen2-VL、InternVL 等）

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/mindspore-lab/mindnlp.git
cd mindnlp/mindnlp-ocr

# 安装依赖
pip install -r requirements.txt

# 或使用 conda
conda create -n vlm-ocr python=3.9
conda activate vlm-ocr
pip install -r requirements.txt
```

### Python SDK 使用

```python
from core.engine import VLMOCREngine
from api.schemas.request import OCRRequest

# 初始化引擎
engine = VLMOCREngine(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda"  # 或 "cpu"
)

# 单图像 OCR
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

request = OCRRequest(
    image=image_bytes,
    output_format="json",  # 或 "text", "markdown"
    language="zh",         # 或 "en", "ja", "ko", "multi"
    task_type="general",   # 或 "document", "table", "formula"
    confidence_threshold=0.8
)

response = engine.predict(request)
print(response.texts)
print(response.boxes)
print(response.confidences)
```

### 启动 API 服务

```bash
# 开发模式
python -m api.app

# 生产模式
uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000
```

访问 API 文档：http://localhost:8000/docs

### API 调用示例

```python
import requests

# 单图像 OCR
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ocr/predict",
        files={"file": f},
        data={
            "output_format": "json",
            "language": "zh",
            "confidence_threshold": 0.8
        }
    )

result = response.json()
print(result)

# 批量 OCR
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb"))
]
response = requests.post(
    "http://localhost:8000/api/v1/ocr/predict_batch",
    files=files,
    data={"output_format": "json", "language": "zh"}
)

# URL OCR
response = requests.post(
    "http://localhost:8000/api/v1/ocr/predict_url",
    json={
        "image_url": "https://example.com/image.jpg",
        "output_format": "json",
        "language": "en"
    }
)
```

## 架构设计

```
mindnlp-ocr/
├── api/                    # API 服务层
│   ├── app.py             # FastAPI 应用入口
│   ├── routes/            # API 路由
│   ├── schemas/           # 请求/响应模型
│   └── middleware/        # 中间件（日志、错误处理）
├── core/                   # 核心业务逻辑
│   ├── engine.py          # 主引擎
│   ├── processor/         # 预处理组件
│   │   ├── image.py       # 图像处理
│   │   ├── prompt.py      # Prompt 构建
│   │   └── batch.py       # 批处理
│   ├── parser/            # 后处理组件
│   │   ├── decoder.py     # Token 解码
│   │   ├── result.py      # 结果解析
│   │   ├── formatter.py   # 输出格式化
│   │   └── result_data.py # 数据类
│   └── validator/         # 输入验证
├── models/                 # 模型层
│   ├── base.py            # 模型基类
│   ├── qwen2vl.py         # Qwen2-VL
│   ├── internvl.py        # InternVL
│   └── loader.py          # 模型加载器
├── config/                 # 配置
├── utils/                  # 工具函数
└── tests/                  # 测试
```

## 组件说明

### 1. API 服务层 (api/)

提供 RESTful API 接口，基于 FastAPI 框架。

**主要端点**：
- `GET /api/v1/health` - 健康检查
- `GET /api/v1/health/ready` - 就绪检查
- `POST /api/v1/ocr/predict` - 单图像 OCR
- `POST /api/v1/ocr/predict_batch` - 批量 OCR
- `POST /api/v1/ocr/predict_url` - URL 图像 OCR

### 2. 核心引擎 (core/engine.py)

协调所有组件，实现端到端的 OCR 流程：

1. 输入验证
2. 图像预处理
3. Prompt 构建
4. 模型推理
5. Token 解码
6. 结果解析
7. 输出格式化

### 3. 预处理组件 (core/processor/)

#### ImageProcessor
- 支持多种输入格式（bytes、PIL、numpy、路径）
- 智能缩放（保持宽高比）
- 自适应 Padding
- ImageNet 归一化
- 记录完整的变换信息（用于坐标还原）

#### PromptBuilder
- 预置模板（通用、文档、表格、公式）
- 多语言支持（中英日韩）
- 多输出格式（JSON、Text、Markdown）
- 自定义 Prompt

#### BatchCollator
- 按图像尺寸动态分组
- 智能 Padding（最小化浪费）
- GPU 优化（32像素对齐）

### 4. 后处理组件 (core/parser/)

#### TokenDecoder
- Token ID → 文本转换
- 批量解码支持
- 输出验证

#### ResultParser
- 多格式解析（JSON、Text、Markdown）
- JSON 容错（代码块、数组格式）
- 置信度过滤
- 解析失败自动降级

#### OutputFormatter
- 坐标映射（还原到原始图像）
- 置信度过滤
- IoU 去重
- 结果排序
- 合并相邻文本块

### 5. 模型层 (models/)

支持多种 VLM 模型：
- **Qwen2-VL**: 通义千问视觉语言模型
- **InternVL**: 书生浦语视觉语言模型
- 可扩展支持其他模型

## 配置

### 环境变量

```bash
# 模型配置
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
DEVICE=cuda

# API 配置
API_HOST=0.0.0.0
API_PORT=8000

# 日志配置
LOG_LEVEL=INFO
```

### 配置文件 (config/settings.py)

```python
from config import Settings

settings = Settings(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda",
    api_host="0.0.0.0",
    api_port=8000,
    log_level="INFO"
)
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_integration.py -v

# 运行带覆盖率的测试
pytest tests/ --cov=core --cov=api --cov-report=html
```

### 测试覆盖

- **单元测试** (tests/test_*.py)
  - 预处理组件：24/24 通过
  - 后处理组件：38/38 通过
  - 模型层：24/24 通过
  - API 层：8/8 通过

- **集成测试** (tests/test_integration.py)
  - 引擎集成测试
  - 组件集成测试
  - 端到端流程测试
  - 错误处理测试
  - 性能测试
  - 多语言测试
  - 输出格式测试

## 性能指标

### 单图像推理

| 模型 | 设备 | 图像尺寸 | 推理时间 | 内存占用 |
|------|------|---------|---------|---------|
| Qwen2-VL-2B | GPU (A100) | 448×448 | ~0.5s | ~4GB |
| Qwen2-VL-2B | CPU | 448×448 | ~3s | ~2GB |
| InternVL-2B | GPU (A100) | 448×448 | ~0.6s | ~5GB |

### 批量处理

| 批次大小 | 设备 | 总时间 | 平均单图 |
|----------|------|--------|----------|
| 1 | GPU | 0.5s | 0.5s |
| 4 | GPU | 1.2s | 0.3s |
| 8 | GPU | 2.0s | 0.25s |

## 示例场景

### 1. 通用 OCR

```python
request = OCRRequest(
    image=image_bytes,
    output_format="json",
    language="zh",
    task_type="general"
)
response = engine.predict(request)

# 输出
{
    "success": true,
    "texts": ["文本内容"],
    "boxes": [[x1, y1, x2, y2]],
    "confidences": [0.95],
    "inference_time": 0.5
}
```

### 2. 文档理解

```python
request = OCRRequest(
    image=document_image,
    output_format="markdown",
    language="zh",
    task_type="document"
)
response = engine.predict(request)

# 输出 Markdown 格式的文档结构
# 标题
段落内容...
## 子标题
更多内容...
```

### 3. 表格识别

```python
request = OCRRequest(
    image=table_image,
    output_format="json",
    language="en",
    task_type="table"
)
response = engine.predict(request)

# 输出结构化的表格数据
{
    "table": {
        "headers": ["列1", "列2"],
        "rows": [
            ["数据1", "数据2"],
            ["数据3", "数据4"]
        ]
    }
}
```

### 4. 公式识别

```python
request = OCRRequest(
    image=formula_image,
    output_format="text",
    language="en",
    task_type="formula"
)
response = engine.predict(request)

# 输出 LaTeX 格式
E = mc^2
```

## 常见问题

### Q1: 如何选择合适的模型？

A: 
- **Qwen2-VL-2B**: 平衡性能和精度，适合大多数场景
- **Qwen2-VL-7B**: 更高精度，需要更多资源
- **InternVL**: 特定场景下的替代选择

### Q2: GPU 内存不足怎么办？

A:
```python
# 使用 CPU
engine = VLMOCREngine(device="cpu")

# 或使用更小的模型
engine = VLMOCREngine(model_name="Qwen/Qwen2-VL-2B-Instruct")

# 或减少批次大小
```

### Q3: 如何提高识别精度？

A:
1. 使用更大的模型
2. 调整置信度阈值
3. 使用自定义 Prompt
4. 预处理图像（去噪、增强对比度）

### Q4: 支持哪些图像格式？

A: 支持 JPEG、PNG、BMP、TIFF、WebP 等常见格式

### Q5: 如何处理超大图像？

A:
```python
# 自动缩放到合适尺寸
processor = ImageProcessor(target_size=(448, 448))

# 或使用切片处理
# （待实现）
```

## 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

Apache 2.0 License

## 致谢

- MindSpore Team
- Transformers Library
- Qwen Team
- InternVL Team

## 联系方式

- Issue Tracker: https://github.com/mindspore-lab/mindnlp/issues
- Documentation: https://github.com/mindspore-lab/mindnlp/blob/master/mindnlp-ocr/README.md
