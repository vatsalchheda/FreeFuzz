import paddle
arg_1_tensor = paddle.rand([5, 20], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-2048, 32, [3, 224, 224, 1], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
res = paddle.nn.functional.dice_loss(input=arg_1,label=arg_2,)
