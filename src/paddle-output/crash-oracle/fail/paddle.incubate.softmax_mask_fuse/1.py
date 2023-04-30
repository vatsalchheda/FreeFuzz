import paddle
arg_1_tensor = paddle.randint(-2048,16,[2, 8, 8, 32], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128,256,[2, 1, 8, 32], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.incubate.softmax_mask_fuse(arg_1,arg_2,)
