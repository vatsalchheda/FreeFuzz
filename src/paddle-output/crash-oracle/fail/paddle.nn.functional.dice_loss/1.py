import paddle
arg_1_tensor = paddle.rand([3, 224, 224, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096, 2048, [3, 224, 224, 1], dtype=paddle.int64arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.dice_loss(input=arg_1,label=arg_2,)
