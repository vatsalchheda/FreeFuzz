import paddle
arg_1_tensor = paddle.randint(-32768,16384,[2, 4], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4,4096,[4], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
res = paddle.bucketize(arg_1,arg_2,)
