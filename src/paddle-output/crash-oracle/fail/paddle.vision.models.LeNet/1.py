import paddle
arg_class = paddle.vision.models.LeNet()
arg_1_0_tensor = paddle.randint(0,2,[90, 0, 54, 28])
arg_1_0 = arg_1_0_tensor.clone()
arg_1 = [arg_1_0,]
res = arg_class(*arg_1)
