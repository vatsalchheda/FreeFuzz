import paddle
arg_1_tensor = paddle.randint(-32768,2,[2, 8, 8, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-8192,1024,[2, 1, 8, 32], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.softmax_mask_fuse(arg_1,arg_2,)
