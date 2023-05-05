import paddle
arg_1_tensor = paddle.randint(-16, 64, [64, 32, 32], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-128, 16, [64, 32, 32], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
arg_3 = 5
res = paddle.fluid.layers.nn.mean_iou(arg_1,arg_2,arg_3,)
