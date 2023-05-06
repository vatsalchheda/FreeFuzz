import paddle
arg_1 = 18
arg_2 = -105.0
res = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
