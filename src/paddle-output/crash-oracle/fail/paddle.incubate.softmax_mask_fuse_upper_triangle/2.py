import paddle
arg_1_tensor = paddle.randint(-1024,128,[48, 1, 32, 0], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.incubate.softmax_mask_fuse_upper_triangle(arg_1,)
