# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
if dist.get_rank() == 0:
    data1 = paddle.to_tensor([7, 8, 9])
    data2 = paddle.to_tensor([10, 11, 12])
    dist.stream.scatter(data1, src=1)
else:
    data1 = paddle.to_tensor([1, 2, 3])
    data2 = paddle.to_tensor([4, 5, 6])
    dist.stream.scatter(data1, [data1, data2], src=1)
out = data1.numpy()
# [1, 2, 3] (2 GPUs, out for rank 0)
# [4, 5, 6] (2 GPUs, out for rank 1)