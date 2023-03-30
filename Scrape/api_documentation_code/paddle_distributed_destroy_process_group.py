# required: distributed
import paddle
import paddle.distributed as dist

dist.init_parallel_env()
group = dist.new_group([0, 1])

dist.destroy_process_group(group)
print(dist.is_initialized())
# True
dist.destroy_process_group()
print(dist.is_initialized())
# False