# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
local_rank = dist.get_rank()

# case 1
output = paddle.empty([2], dtype="int64")
if local_rank == 0:
    data = paddle.to_tensor([0, 1])
else:
    data = paddle.to_tensor([2, 3])
task = dist.stream.alltoall_single(output, data, sync_op=False)
task.wait()
out = output.numpy()
# [0, 2] (2 GPUs, out for rank 0)
# [1, 3] (2 GPUs, out for rank 1)

# case 2
size = dist.get_world_size()
output = paddle.empty([(local_rank + 1) * size, size], dtype='float32')
if local_rank == 0:
    data = paddle.to_tensor([[0., 0.], [0., 0.], [0., 0.]])
else:
    data = paddle.to_tensor([[1., 1.], [1., 1.], [1., 1.]])
out_split_sizes = [local_rank + 1 for i in range(size)]
in_split_sizes = [i + 1 for i in range(size)]
task = dist.stream.alltoall_single(output,
                                data,
                                out_split_sizes,
                                in_split_sizes,
                                sync_op=False)
task.wait()
out = output.numpy()
# [[0., 0.], [1., 1.]]                     (2 GPUs, out for rank 0)
# [[0., 0.], [0., 0.], [1., 1.], [1., 1.]] (2 GPUs, out for rank 1)