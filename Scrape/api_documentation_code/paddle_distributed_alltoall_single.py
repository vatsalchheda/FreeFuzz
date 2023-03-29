# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
rank = dist.get_rank()
size = dist.get_world_size()

# case 1 (2 GPUs)
data = paddle.arange(2, dtype='int64') + rank * 2
# data for rank 0: [0, 1]
# data for rank 1: [2, 3]
output = paddle.empty([2], dtype='int64')
dist.alltoall_single(data, output)
print(output)
# output for rank 0: [0, 2]
# output for rank 1: [1, 3]

# case 2 (2 GPUs)
in_split_sizes = [i + 1 for i in range(size)]
# in_split_sizes for rank 0: [1, 2]
# in_split_sizes for rank 1: [1, 2]
out_split_sizes = [rank + 1 for i in range(size)]
# out_split_sizes for rank 0: [1, 1]
# out_split_sizes for rank 1: [2, 2]
data = paddle.ones([sum(in_split_sizes), size], dtype='float32') * rank
# data for rank 0: [[0., 0.], [0., 0.], [0., 0.]]
# data for rank 1: [[1., 1.], [1., 1.], [1., 1.]]
output = paddle.empty([(rank + 1) * size, size], dtype='float32')
group = dist.new_group([0, 1])
task = dist.alltoall_single(data,
                            output,
                            in_split_sizes,
                            out_split_sizes,
                            sync_op=False,
                            group=group)
task.wait()
print(output)
# output for rank 0: [[0., 0.], [1., 1.]]
# output for rank 1: [[0., 0.], [0., 0.], [1., 1.], [1., 1.]]