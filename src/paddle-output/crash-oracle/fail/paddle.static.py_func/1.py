import paddle
arg_1 = "debug_func"
arg_2_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
res = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,)
