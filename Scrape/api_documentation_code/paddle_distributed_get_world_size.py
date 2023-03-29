# Execute this script using distributed launch with one card configs.
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
print("The world_size is %d" % dist.get_world_size())
# The world_size is 1