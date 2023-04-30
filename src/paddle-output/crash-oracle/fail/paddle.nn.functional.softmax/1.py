import paddle
arg_1_tensor = paddle.randint(-1,16,[3, 224, 224, 2], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.nn.functional.softmax(arg_1,)
