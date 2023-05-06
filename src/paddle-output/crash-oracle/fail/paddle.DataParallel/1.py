import paddle
arg_1 = "__main__SimpleNet"
arg_class = paddle.DataParallel(arg_1,)
arg_2_0_tensor = paddle.rand([10, 10], dtype=paddle.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
res = arg_class(*arg_2)
