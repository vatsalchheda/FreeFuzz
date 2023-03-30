# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
object_list = []
if dist.get_rank() == 0:
    obj = {"foo": [1, 2, 3]}
else:
    obj = {"bar": [4, 5, 6]}
dist.all_gather_object(object_list, obj)
print(object_list)
# [{'foo': [1, 2, 3]}, {'bar': [4, 5, 6]}] (2 GPUs)