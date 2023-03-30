# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
if dist.get_rank() == 0:
    data = paddle.to_tensor([7, 8, 9])
    dist.send(data, dst=1)
else:
    data = paddle.to_tensor([1, 2, 3])
    dist.recv(data, src=0)
print(data)
# [7, 8, 9] (2 GPUs)