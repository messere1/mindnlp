"""
批处理优化测试
"""

import pytest
import pytest_asyncio
import torch
import numpy as np
from PIL import Image
import asyncio
import time
from core.processor.batch import AdaptiveBatchCollator
from core.batch_queue import AsyncBatchQueue, BatchQueueManager
from core.metrics import BatchPerformanceMonitor


class TestAdaptiveBatchCollator:
    """自适应批处理整理器测试"""
    
    @pytest.fixture
    def collator(self):
        return AdaptiveBatchCollator(
            max_batch_size=8,
            size_threshold=0.2,
            enable_dynamic_batching=True
        )
    
    @pytest.fixture
    def sample_images(self):
        """生成不同尺寸的测试图像"""
        return [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (120, 110), color='blue'),
            Image.new('RGB', (200, 200), color='green'),
            Image.new('RGB', (210, 195), color='yellow'),
            Image.new('RGB', (300, 150), color='purple'),
            Image.new('RGB', (310, 160), color='orange'),
        ]
    
    def test_group_by_size(self, collator, sample_images):
        """测试按尺寸分组"""
        groups = collator.group_by_size(sample_images)
        
        assert len(groups) > 0, "应该至少有一个组"
        assert len(groups) <= len(sample_images), "组数不应超过图像数"
        
        # 验证所有图像都被分组
        total_images = sum(len(group) for group in groups)
        assert total_images == len(sample_images), "所有图像都应被分组"
        
        print(f"Grouped {len(sample_images)} images into {len(groups)} groups")
        for i, group in enumerate(groups):
            print(f"  Group {i+1}: {len(group)} images")
    
    def test_group_by_size_with_limit(self, collator, sample_images):
        """测试带限制的分组"""
        max_groups = 2
        groups = collator.group_by_size(sample_images, max_groups=max_groups)
        
        assert len(groups) <= max_groups, f"组数应不超过{max_groups}"
    
    def test_dynamic_padding(self, collator):
        """测试动态填充"""
        # 创建不同尺寸的tensor
        tensors = [
            torch.randn(3, 100, 100),
            torch.randn(3, 150, 120),
            torch.randn(3, 200, 180),
        ]
        
        padded_batch, padding_info = collator.dynamic_padding(tensors, align_to=32)
        
        # 验证批次形状
        assert padded_batch.shape[0] == len(tensors), "批大小应匹配"
        assert padded_batch.shape[1] == 3, "通道数应为3"
        
        # 验证对齐
        assert padded_batch.shape[2] % 32 == 0, "高度应对齐到32"
        assert padded_batch.shape[3] % 32 == 0, "宽度应对齐到32"
        
        # 验证padding_info
        assert 'padded_size' in padding_info
        assert 'original_shapes' in padding_info
        assert len(padding_info['original_shapes']) == len(tensors)
        
        print(f"Padded to: {padded_batch.shape}")
        print(f"Padding ratio: {padding_info['padding_ratio']:.2f}")
    
    def test_get_optimal_batch_size(self, collator):
        """测试最优批大小计算"""
        image_sizes = [
            (224, 224),
            (256, 256),
            (512, 512),
        ]
        
        # 模拟1GB可用内存
        available_memory = 1 * 1024**3
        
        optimal_size = collator.get_optimal_batch_size(
            image_sizes,
            available_memory=available_memory
        )
        
        assert optimal_size > 0, "最优批大小应大于0"
        assert optimal_size <= collator.max_batch_size, "不应超过最大批大小"
        
        print(f"Optimal batch size: {optimal_size}")
    
    def test_collate_adaptive(self, collator, sample_images):
        """测试自适应整理"""
        batches = collator.collate_adaptive(sample_images, auto_group=True)
        
        assert len(batches) > 0, "应该至少有一个批次"
        
        for batch_tensor, batch_info in batches:
            assert isinstance(batch_tensor, torch.Tensor), "应返回Tensor"
            assert 'group_id' in batch_info
            assert 'batch_size' in batch_info
            assert 'padding_info' in batch_info
            
            print(f"Batch {batch_info['group_id']}: size={batch_info['batch_size']}, "
                  f"shape={batch_tensor.shape}")


class TestAsyncBatchQueue:
    """异步批处理队列测试"""
    
    @pytest_asyncio.fixture
    async def queue(self):
        """创建测试队列"""
        def process_fn(batch_data):
            """模拟批处理函数"""
            time.sleep(0.1)  # 模拟处理时间
            return [f"result_{i}" for i in range(len(batch_data))]
        
        queue = AsyncBatchQueue(
            process_fn=process_fn,
            max_batch_size=4,
            max_wait_time=0.2,
            max_queue_size=20
        )
        
        await queue.start()
        yield queue
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_single_request(self, queue):
        """测试单个请求"""
        result = await queue.add_request("req-1", {"data": "test"}, timeout=5.0)
        
        assert result is not None
        assert isinstance(result, str)
        print(f"Single request result: {result}")
    
    @pytest.mark.asyncio
    async def test_multiple_requests(self, queue):
        """测试多个请求"""
        tasks = []
        for i in range(10):
            task = queue.add_request(f"req-{i}", {"data": f"test-{i}"}, timeout=10.0)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10, "应返回10个结果"
        print(f"Multiple requests completed: {len(results)} results")
    
    @pytest.mark.asyncio
    async def test_batching_aggregation(self, queue):
        """测试批处理聚合"""
        # 快速提交多个请求
        tasks = []
        for i in range(8):
            task = queue.add_request(f"req-{i}", {"data": f"test-{i}"}, timeout=10.0)
            tasks.append(task)
            await asyncio.sleep(0.01)  # 小延迟，让请求聚合
        
        results = await asyncio.gather(*tasks)
        
        metrics = queue.get_metrics()
        print(f"Batching metrics: {metrics}")
        print(f"Avg batch size: {metrics['avg_batch_size']:.2f}")
        
        assert metrics['total_requests'] == 8
        assert metrics['avg_batch_size'] > 1.0, "应该有批处理聚合"
    
    @pytest.mark.asyncio
    async def test_queue_metrics(self, queue):
        """测试队列指标"""
        # 提交一些请求
        tasks = []
        for i in range(5):
            task = queue.add_request(f"req-{i}", {"data": f"test-{i}"}, timeout=10.0)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        metrics = queue.get_metrics()
        
        assert 'total_requests' in metrics
        assert 'total_batches' in metrics
        assert 'avg_batch_size' in metrics
        assert 'avg_wait_time' in metrics
        
        print(f"Queue metrics: {metrics}")
    
    @pytest.mark.asyncio
    async def test_timeout(self):
        """测试超时"""
        def slow_process_fn(batch_data):
            time.sleep(5)  # 很慢的处理
            return [f"result_{i}" for i in range(len(batch_data))]
        
        queue = AsyncBatchQueue(
            process_fn=slow_process_fn,
            max_batch_size=4,
            max_wait_time=0.1
        )
        
        await queue.start()
        
        try:
            with pytest.raises(asyncio.TimeoutError):
                await queue.add_request("req-timeout", {"data": "test"}, timeout=1.0)
        finally:
            await queue.stop()


class TestBatchQueueManager:
    """批处理队列管理器测试"""
    
    @pytest.fixture
    def manager(self):
        return BatchQueueManager()
    
    def test_create_queue(self, manager):
        """测试创建队列"""
        def dummy_fn(batch_data):
            return batch_data
        
        queue = manager.create_queue("test-queue", dummy_fn, max_batch_size=8)
        
        assert queue is not None
        assert manager.get_queue("test-queue") is queue
    
    def test_duplicate_queue(self, manager):
        """测试重复创建队列"""
        def dummy_fn(batch_data):
            return batch_data
        
        manager.create_queue("test-queue", dummy_fn)
        
        with pytest.raises(ValueError):
            manager.create_queue("test-queue", dummy_fn)
    
    @pytest.mark.asyncio
    async def test_start_stop_all(self, manager):
        """测试启动/停止所有队列"""
        def dummy_fn(batch_data):
            return batch_data
        
        manager.create_queue("queue-1", dummy_fn)
        manager.create_queue("queue-2", dummy_fn)
        
        await manager.start_all()
        
        # 验证队列都在运行
        assert manager.get_queue("queue-1").is_running
        assert manager.get_queue("queue-2").is_running
        
        await manager.stop_all()
        
        # 验证队列都已停止
        assert not manager.get_queue("queue-1").is_running
        assert not manager.get_queue("queue-2").is_running


class TestBatchPerformanceMonitor:
    """批处理性能监控测试"""
    
    @pytest.fixture
    def monitor(self):
        return BatchPerformanceMonitor(enable_gpu_monitoring=False)
    
    def test_start_end_batch(self, monitor):
        """测试批处理计时"""
        start_time = monitor.start_batch()
        time.sleep(0.1)  # 模拟处理
        metrics = monitor.end_batch(start_time, batch_size=8, padding_ratio=1.2)
        
        assert metrics.batch_size == 8
        assert metrics.processing_time >= 0.1
        assert metrics.throughput > 0
        assert metrics.padding_ratio == 1.2
        
        print(f"Batch metrics: {metrics}")
    
    def test_get_current_metrics(self, monitor):
        """测试获取当前指标"""
        # 记录一些批次
        for i in range(3):
            start = monitor.start_batch()
            time.sleep(0.05)
            monitor.end_batch(start, batch_size=4 + i)
        
        current = monitor.get_current_metrics()
        
        assert 'batch_size' in current
        assert 'processing_time' in current
        assert 'throughput' in current
        
        print(f"Current metrics: {current}")
    
    def test_get_aggregate_metrics(self, monitor):
        """测试获取聚合指标"""
        # 记录多个批次
        for i in range(10):
            start = monitor.start_batch()
            time.sleep(0.02)
            monitor.end_batch(start, batch_size=4 + (i % 3))
        
        aggregate = monitor.get_aggregate_metrics()
        
        assert 'avg_batch_size' in aggregate
        assert 'avg_throughput' in aggregate
        assert 'total_batches' in aggregate
        assert aggregate['total_batches'] == 10
        
        print(f"Aggregate metrics: {aggregate}")
    
    def test_performance_report(self, monitor):
        """测试性能报告"""
        # 记录一些批次
        for i in range(5):
            start = monitor.start_batch()
            time.sleep(0.05)
            monitor.end_batch(start, batch_size=8)
        
        report = monitor.get_performance_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        print("\n" + report)
    
    def test_reset(self, monitor):
        """测试重置指标"""
        # 记录一些批次
        for i in range(3):
            start = monitor.start_batch()
            monitor.end_batch(start, batch_size=4)
        
        assert monitor.total_batches == 3
        
        monitor.reset()
        
        assert monitor.total_batches == 0
        assert monitor.total_images == 0
        assert len(monitor.metrics_history) == 0


@pytest.mark.performance
class TestBatchPerformance:
    """批处理性能测试"""
    
    @pytest.fixture
    def collator(self):
        return AdaptiveBatchCollator(max_batch_size=16)
    
    @pytest.fixture
    def monitor(self):
        return BatchPerformanceMonitor(enable_gpu_monitoring=False)
    
    def test_throughput_improvement(self, collator, monitor):
        """测试吞吐量提升"""
        # 生成大量测试图像
        images = [
            torch.randn(3, 224 + i * 10, 224 + i * 10)
            for i in range(32)
        ]
        
        # 不使用批处理（逐个处理）
        start = monitor.start_batch()
        for img in images:
            # 模拟处理
            _ = img * 2
        sequential_metrics = monitor.end_batch(start, batch_size=len(images))
        
        # 使用自适应批处理
        start = monitor.start_batch()
        batches = collator.collate_adaptive(images, auto_group=True)
        for batch_tensor, _ in batches:
            # 模拟批处理
            _ = batch_tensor * 2
        batched_metrics = monitor.end_batch(start, batch_size=len(images))
        
        print(f"\nSequential throughput: {sequential_metrics.throughput:.2f} images/s")
        print(f"Batched throughput: {batched_metrics.throughput:.2f} images/s")
        print(f"Improvement: {batched_metrics.throughput / sequential_metrics.throughput:.2f}x")
        
        # 批处理应该更快
        assert batched_metrics.throughput >= sequential_metrics.throughput * 0.8, \
            "批处理吞吐量应该接近或超过顺序处理"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
