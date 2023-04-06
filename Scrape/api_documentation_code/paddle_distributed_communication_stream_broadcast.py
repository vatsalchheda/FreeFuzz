# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
local_rank = dist.get_rank()
if local_rank == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
task = dist.stream.broadcast(data, src=1, sync_op=False)
task.wait()
out = data.numpy()
# [[1, 2, 3], [1, 2, 3]] (2 GPUs)