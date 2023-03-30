# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
if dist.get_rank() == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
dist.reduce(data, dst=0)
print(data)
# [[5, 7, 9], [5, 7, 9]] (2 GPUs, out for rank 0)
# [[1, 2, 3], [1, 2, 3]] (2 GPUs, out for rank 1)