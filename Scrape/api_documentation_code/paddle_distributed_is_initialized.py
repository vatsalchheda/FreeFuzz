# required: distributed
import paddle

print(paddle.distributed.is_initialized())
# False

paddle.distributed.init_parallel_env()
print(paddle.distributed.is_initialized())
# True