import paddle
from paddle.distributed import init_parallel_env

paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
init_parallel_env()
paddle.distributed.barrier()