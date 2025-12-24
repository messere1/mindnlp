"""
批处理性能监控
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class BatchMetrics:
    """批处理指标"""
    batch_size: int = 0
    processing_time: float = 0.0
    throughput: float = 0.0  # images/second
    gpu_memory_used: float = 0.0  # GB
    gpu_memory_total: float = 0.0  # GB
    gpu_utilization: float = 0.0  # %
    padding_ratio: float = 0.0  # 填充浪费比例
    timestamp: float = field(default_factory=time.time)


class BatchPerformanceMonitor:
    """
    批处理性能监控器
    
    功能:
    1. 批处理性能指标收集
    2. GPU资源监控
    3. 吞吐量统计
    4. 性能趋势分析
    """
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        """
        初始化性能监控器
        
        Args:
            enable_gpu_monitoring: 是否启用GPU监控
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # 指标历史
        self.metrics_history = []
        self.max_history_size = 1000
        
        # 统计信息
        self.total_batches = 0
        self.total_images = 0
        self.total_time = 0.0
        
        logger.info(f"BatchPerformanceMonitor initialized (GPU monitoring: {self.enable_gpu_monitoring})")
    
    def start_batch(self) -> float:
        """
        开始批处理计时
        
        Returns:
            start_time: 开始时间戳
        """
        return time.time()
    
    def end_batch(
        self,
        start_time: float,
        batch_size: int,
        padding_ratio: Optional[float] = None
    ) -> BatchMetrics:
        """
        结束批处理并记录指标
        
        Args:
            start_time: 开始时间
            batch_size: 批大小
            padding_ratio: 填充浪费比例
            
        Returns:
            metrics: 批处理指标
        """
        processing_time = time.time() - start_time
        throughput = batch_size / processing_time if processing_time > 0 else 0.0
        
        # GPU指标
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_utilization = 0.0
        
        if self.enable_gpu_monitoring:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                # 尝试获取GPU利用率（需要nvidia-ml-py3）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu
                    pynvml.nvmlShutdown()
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
        
        # 创建指标对象
        metrics = BatchMetrics(
            batch_size=batch_size,
            processing_time=processing_time,
            throughput=throughput,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            padding_ratio=padding_ratio or 0.0
        )
        
        # 记录指标
        self._record_metrics(metrics)
        
        return metrics
    
    def _record_metrics(self, metrics: BatchMetrics):
        """记录指标到历史"""
        self.metrics_history.append(metrics)
        
        # 限制历史大小
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
        
        # 更新统计
        self.total_batches += 1
        self.total_images += metrics.batch_size
        self.total_time += metrics.processing_time
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        
        return {
            'batch_size': latest.batch_size,
            'processing_time': latest.processing_time,
            'throughput': latest.throughput,
            'gpu_memory_used': latest.gpu_memory_used,
            'gpu_memory_total': latest.gpu_memory_total,
            'gpu_memory_percent': (
                latest.gpu_memory_used / latest.gpu_memory_total * 100
                if latest.gpu_memory_total > 0 else 0.0
            ),
            'gpu_utilization': latest.gpu_utilization,
            'padding_ratio': latest.padding_ratio
        }
    
    def get_aggregate_metrics(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        获取聚合指标
        
        Args:
            window_size: 窗口大小（None表示全部历史）
            
        Returns:
            aggregate_metrics: 聚合指标
        """
        if not self.metrics_history:
            return {}
        
        # 获取窗口数据
        if window_size:
            metrics_window = self.metrics_history[-window_size:]
        else:
            metrics_window = self.metrics_history
        
        # 计算聚合统计
        avg_batch_size = sum(m.batch_size for m in metrics_window) / len(metrics_window)
        avg_processing_time = sum(m.processing_time for m in metrics_window) / len(metrics_window)
        avg_throughput = sum(m.throughput for m in metrics_window) / len(metrics_window)
        avg_gpu_memory = sum(m.gpu_memory_used for m in metrics_window) / len(metrics_window)
        avg_gpu_util = sum(m.gpu_utilization for m in metrics_window) / len(metrics_window)
        avg_padding_ratio = sum(m.padding_ratio for m in metrics_window) / len(metrics_window)
        
        # 最大/最小值
        max_throughput = max(m.throughput for m in metrics_window)
        min_throughput = min(m.throughput for m in metrics_window)
        max_gpu_memory = max(m.gpu_memory_used for m in metrics_window)
        
        return {
            'window_size': len(metrics_window),
            'avg_batch_size': avg_batch_size,
            'avg_processing_time': avg_processing_time,
            'avg_throughput': avg_throughput,
            'max_throughput': max_throughput,
            'min_throughput': min_throughput,
            'avg_gpu_memory_used': avg_gpu_memory,
            'max_gpu_memory_used': max_gpu_memory,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_padding_ratio': avg_padding_ratio,
            'total_batches': self.total_batches,
            'total_images': self.total_images,
            'total_time': self.total_time,
            'overall_throughput': (
                self.total_images / self.total_time if self.total_time > 0 else 0.0
            )
        }
    
    def get_performance_report(self) -> str:
        """
        生成性能报告
        
        Returns:
            report: 性能报告字符串
        """
        if not self.metrics_history:
            return "No metrics available"
        
        current = self.get_current_metrics()
        aggregate = self.get_aggregate_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           Batch Processing Performance Report               ║
╠══════════════════════════════════════════════════════════════╣
║ Overall Statistics:                                          ║
║   Total Batches:        {aggregate['total_batches']:>6}                            ║
║   Total Images:         {aggregate['total_images']:>6}                            ║
║   Total Time:           {aggregate['total_time']:>6.2f}s                           ║
║   Overall Throughput:   {aggregate['overall_throughput']:>6.2f} images/s                  ║
╠══════════════════════════════════════════════════════════════╣
║ Average Metrics:                                             ║
║   Avg Batch Size:       {aggregate['avg_batch_size']:>6.2f}                           ║
║   Avg Processing Time:  {aggregate['avg_processing_time']:>6.3f}s                          ║
║   Avg Throughput:       {aggregate['avg_throughput']:>6.2f} images/s                  ║
║   Max Throughput:       {aggregate['max_throughput']:>6.2f} images/s                  ║
║   Min Throughput:       {aggregate['min_throughput']:>6.2f} images/s                  ║
╠══════════════════════════════════════════════════════════════╣
║ GPU Metrics:                                                 ║
║   Avg GPU Memory:       {aggregate['avg_gpu_memory_used']:>6.2f} GB                       ║
║   Max GPU Memory:       {aggregate['max_gpu_memory_used']:>6.2f} GB                       ║
║   Avg GPU Utilization:  {aggregate['avg_gpu_utilization']:>6.1f}%                         ║
║   Avg Padding Ratio:    {aggregate['avg_padding_ratio']:>6.2f}x                         ║
╠══════════════════════════════════════════════════════════════╣
║ Current Batch:                                               ║
║   Batch Size:           {current.get('batch_size', 0):>6}                            ║
║   Processing Time:      {current.get('processing_time', 0):>6.3f}s                          ║
║   Throughput:           {current.get('throughput', 0):>6.2f} images/s                  ║
║   GPU Memory:           {current.get('gpu_memory_used', 0):>6.2f}/{current.get('gpu_memory_total', 0):>6.2f} GB ({current.get('gpu_memory_percent', 0):>5.1f}%)  ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report
    
    def reset(self):
        """重置所有指标"""
        self.metrics_history.clear()
        self.total_batches = 0
        self.total_images = 0
        self.total_time = 0.0
        logger.info("Performance metrics reset")
