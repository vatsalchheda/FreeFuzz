import paddle
arg_1 = 36
arg_2 = 0
res = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
