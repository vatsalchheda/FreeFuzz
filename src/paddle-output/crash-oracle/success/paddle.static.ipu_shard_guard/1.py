import paddle
arg_1 = -43
arg_2 = -16
res = paddle.static.ipu_shard_guard(index=arg_1,stage=arg_2,)
