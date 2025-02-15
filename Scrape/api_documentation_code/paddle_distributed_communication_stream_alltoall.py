# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
out_tensor_list = []
if dist.get_rank() == 0:
    data1 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
    data2 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]])
else:
    data1 = paddle.to_tensor([[13, 14, 15], [16, 17, 18]])
    data2 = paddle.to_tensor([[19, 20, 21], [22, 23, 24]])
task = dist.stream.alltoall(out_tensor_list, [data1, data2], sync_op=False)
task.wait()
print(out_tensor_list)
# [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]]    (2 GPUs, out for rank 0)
# [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]] (2 GPUs, out for rank 1)