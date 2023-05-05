import paddle
arg_1 = "__mp_main__LinearNet"
arg_2 = None
arg_class = paddle.DataParallel(arg_1,group=arg_2,)
arg_3_0_tensor = paddle.rand([10, 10], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
