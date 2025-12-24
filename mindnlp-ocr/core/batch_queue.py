"""
异步批处理队列
支持请求聚合和非阻塞式批处理
"""

import asyncio
import time
from typing import Any, Callable, List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import deque
import torch
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class BatchRequest:
    """批处理请求"""
    request_id: str
    data: Any
    future: asyncio.Future
    timestamp: float


class AsyncBatchQueue:
    """
    异步批处理队列
    
    特性:
    1. 自动批处理聚合
    2. 智能等待策略（时间/数量触发）
    3. 非阻塞式处理
    4. 背压控制（防止队列溢出）
    """
    
    def __init__(
        self,
        process_fn: Callable,
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,
        max_queue_size: int = 100,
        enable_metrics: bool = True
    ):
        """
        初始化异步批处理队列
        
        Args:
            process_fn: 批处理函数，接收List[data]，返回List[result]
            max_batch_size: 最大批大小
            max_wait_time: 最大等待时间（秒）
            max_queue_size: 最大队列大小
            enable_metrics: 是否启用指标统计
        """
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.enable_metrics = enable_metrics
        
        self.queue: deque = deque()
        self.queue_lock = asyncio.Lock()
        self.is_running = False
        self.process_task: Optional[asyncio.Task] = None
        
        # 指标统计
        self.metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_wait_time': 0.0,
            'queue_full_count': 0
        }
        
        logger.info(
            f"AsyncBatchQueue initialized: "
            f"max_batch_size={max_batch_size}, "
            f"max_wait_time={max_wait_time}s"
        )
    
    async def start(self):
        """启动批处理循环"""
        if self.is_running:
            logger.warning("AsyncBatchQueue is already running")
            return
        
        self.is_running = True
        self.process_task = asyncio.create_task(self._process_loop())
        logger.info("AsyncBatchQueue started")
    
    async def stop(self):
        """停止批处理循环"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
        
        # 处理剩余请求
        if self.queue:
            logger.info(f"Processing {len(self.queue)} remaining requests...")
            await self._process_batch(list(self.queue))
            self.queue.clear()
        
        logger.info("AsyncBatchQueue stopped")
    
    async def add_request(
        self,
        request_id: str,
        data: Any,
        timeout: Optional[float] = None
    ) -> Any:
        """
        添加请求到队列
        
        Args:
            request_id: 请求ID
            data: 请求数据
            timeout: 超时时间（秒）
            
        Returns:
            result: 处理结果
        """
        if not self.is_running:
            raise RuntimeError("AsyncBatchQueue is not running. Call start() first.")
        
        # 检查队列是否已满
        async with self.queue_lock:
            if len(self.queue) >= self.max_queue_size:
                if self.enable_metrics:
                    self.metrics['queue_full_count'] += 1
                raise RuntimeError(
                    f"Queue is full (size={self.max_queue_size}). "
                    "Please retry later or increase max_queue_size."
                )
            
            # 创建请求
            future = asyncio.Future()
            request = BatchRequest(
                request_id=request_id,
                data=data,
                future=future,
                timestamp=time.time()
            )
            
            self.queue.append(request)
            
            if self.enable_metrics:
                self.metrics['total_requests'] += 1
        
        # 等待结果
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timeout after {timeout}s")
            raise
    
    async def _process_loop(self):
        """批处理循环"""
        logger.info("Batch processing loop started")
        
        while self.is_running:
            try:
                # 收集批次
                batch = await self._collect_batch()
                
                if batch:
                    # 处理批次
                    await self._process_batch(batch)
                else:
                    # 如果没有请求，短暂休眠
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in process loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """
        收集批次（最多等待 max_wait_time）
        
        Returns:
            batch: 批次请求列表
        """
        batch = []
        deadline = time.time() + self.max_wait_time
        
        while len(batch) < self.max_batch_size:
            # 检查是否超时
            remaining_time = deadline - time.time()
            if remaining_time <= 0 and batch:
                break
            
            # 尝试从队列获取请求
            async with self.queue_lock:
                if self.queue:
                    request = self.queue.popleft()
                    batch.append(request)
                    continue
            
            # 队列为空，等待一小段时间
            if remaining_time > 0:
                await asyncio.sleep(min(0.001, remaining_time))
            else:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """
        处理批次
        
        Args:
            batch: 批次请求列表
        """
        if not batch:
            return
        
        batch_start_time = time.time()
        batch_size = len(batch)
        
        try:
            # 提取数据
            batch_data = [req.data for req in batch]
            request_ids = [req.request_id for req in batch]
            
            logger.debug(f"Processing batch: size={batch_size}, ids={request_ids}")
            
            # 调用批处理函数
            try:
                results = await asyncio.to_thread(self.process_fn, batch_data)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}", exc_info=True)
                # 设置所有请求为失败
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)
                return
            
            # 分发结果
            if len(results) != batch_size:
                logger.error(
                    f"Result count mismatch: expected {batch_size}, got {len(results)}"
                )
                error = ValueError(f"Result count mismatch")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(error)
                return
            
            for req, result in zip(batch, results):
                if not req.future.done():
                    req.future.set_result(result)
            
            # 更新指标
            if self.enable_metrics:
                batch_time = time.time() - batch_start_time
                wait_times = [batch_start_time - req.timestamp for req in batch]
                avg_wait = sum(wait_times) / len(wait_times)
                
                self.metrics['total_batches'] += 1
                total_batches = self.metrics['total_batches']
                
                # 更新移动平均
                self.metrics['avg_batch_size'] = (
                    (self.metrics['avg_batch_size'] * (total_batches - 1) + batch_size)
                    / total_batches
                )
                self.metrics['avg_wait_time'] = (
                    (self.metrics['avg_wait_time'] * (total_batches - 1) + avg_wait)
                    / total_batches
                )
                
                logger.debug(
                    f"Batch processed: size={batch_size}, "
                    f"batch_time={batch_time:.3f}s, avg_wait={avg_wait:.3f}s"
                )
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            # 设置所有未完成的请求为异常
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取队列指标"""
        return {
            **self.metrics,
            'queue_size': len(self.queue),
            'is_running': self.is_running
        }
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'avg_wait_time': 0.0,
            'queue_full_count': 0
        }
        logger.info("Metrics reset")


class BatchQueueManager:
    """
    批处理队列管理器
    管理多个异步批处理队列
    """
    
    def __init__(self):
        self.queues: Dict[str, AsyncBatchQueue] = {}
        logger.info("BatchQueueManager initialized")
    
    def create_queue(
        self,
        name: str,
        process_fn: Callable,
        **kwargs
    ) -> AsyncBatchQueue:
        """
        创建新的批处理队列
        
        Args:
            name: 队列名称
            process_fn: 批处理函数
            **kwargs: AsyncBatchQueue的其他参数
            
        Returns:
            queue: 创建的队列
        """
        if name in self.queues:
            raise ValueError(f"Queue '{name}' already exists")
        
        queue = AsyncBatchQueue(process_fn, **kwargs)
        self.queues[name] = queue
        logger.info(f"Created queue: {name}")
        
        return queue
    
    def get_queue(self, name: str) -> Optional[AsyncBatchQueue]:
        """获取队列"""
        return self.queues.get(name)
    
    async def start_all(self):
        """启动所有队列"""
        for name, queue in self.queues.items():
            await queue.start()
            logger.info(f"Started queue: {name}")
    
    async def stop_all(self):
        """停止所有队列"""
        for name, queue in self.queues.items():
            await queue.stop()
            logger.info(f"Stopped queue: {name}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有队列的指标"""
        return {
            name: queue.get_metrics()
            for name, queue in self.queues.items()
        }
