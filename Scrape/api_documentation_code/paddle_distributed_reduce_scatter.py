# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
if dist.get_rank() == 0:
    data1 = paddle.to_tensor([0, 1])
    data2 = paddle.to_tensor([2, 3])
else:
    data1 = paddle.to_tensor([4, 5])
    data2 = paddle.to_tensor([6, 7])
dist.reduce_scatter(data1, [data1, data2])
print(data1)
# [4, 6] (2 GPUs, out for rank 0)
# [8, 10] (2 GPUs, out for rank 1)