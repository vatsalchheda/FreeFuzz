# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
local_rank = dist.get_rank()
tensor_list = []
if local_rank == 0:
    data = paddle.to_tensor([[4, 5, 6], [4, 5, 6]])
else:
    data = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
task = dist.stream.all_gather(tensor_list, data, sync_op=False)
task.wait()
print(tensor_list)
# [[[4, 5, 6], [4, 5, 6]], [[1, 2, 3], [1, 2, 3]]] (2 GPUs)