import paddle
arg_1_tensor = paddle.randint(-16,512,[-1, 784], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 10
arg_3 = "softmax"
res = paddle.static.nn.fc(arg_1,size=arg_2,activation=arg_3,)
