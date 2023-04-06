import paddle

paddle.distributed.init_parallel_env()
tindata = paddle.randn(shape=[2, 3])
gp = paddle.distributed.new_group([2,4,6])
paddle.distributed.all_reduce(tindata, group=gp, sync_op=False)