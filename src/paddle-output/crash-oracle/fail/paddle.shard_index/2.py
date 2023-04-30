import paddle
arg_1_tensor = paddle.randint(-8192,64,[2, 1], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_3 = 2
arg_4 = 16
res = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
