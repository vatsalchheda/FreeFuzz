import paddle
arg_1_tensor = paddle.randint(-2, 64, [2, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = 20
arg_3 = 2
arg_4 = True
res = paddle.shard_index(input=arg_1,index_num=arg_2,nshards=arg_3,shard_id=arg_4,)
