import paddle
arg_1_tensor = paddle.randint(0,2,[1], dtype=paddle.bool)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-256,2,[1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-32,512,[1], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
res = paddle.where(arg_1,arg_2,arg_3,)
