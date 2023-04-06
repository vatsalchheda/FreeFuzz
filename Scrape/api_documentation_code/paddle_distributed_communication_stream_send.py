# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
local_rank = dist.get_rank()
if local_rank == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
    task = dist.stream.send(data, dst=1, sync_op=False)
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
    task = dist.stream.recv(data, src=0, sync_op=False)
task.wait()
out = data.numpy()
# [[4, 5, 6], [4, 5, 6]] (2 GPUs)