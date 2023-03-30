import paddle
parallel_mode = paddle.distributed.ParallelMode
print(parallel_mode.DATA_PARALLEL)  # 0