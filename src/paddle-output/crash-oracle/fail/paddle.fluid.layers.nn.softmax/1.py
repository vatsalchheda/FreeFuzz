import paddle
arg_1_tensor = paddle.randint(-2048,128,[2, 21, 10], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
res = paddle.fluid.layers.nn.softmax(input=arg_1,)
