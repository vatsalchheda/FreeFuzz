import paddle
arg_1_tensor = paddle.randint(-128,16,[2, 9, 4, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 14
res = paddle.nn.functional.pixel_shuffle(arg_1,arg_2,)
