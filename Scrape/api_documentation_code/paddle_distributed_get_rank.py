# Execute this script using distributed launch with one card configs.
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
print("The rank is %d" % dist.get_rank())
# The rank is 0