import paddle
arg_1_tensor = paddle.randint(-32,2,[3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2,16,[1, 1, 2, 2, 3], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = 20
arg_4 = 0
res = paddle.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,padding=arg_4,)
