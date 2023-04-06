import paddle

paddle.distributed.init_parallel_env()
tindata = paddle.randn(shape=[2, 3])
paddle.distributed.all_reduce(tindata, sync_op=True)
paddle.distributed.wait(tindata)