import paddle
arg_1 = "tanh"
arg_2_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "tanh_grad"
arg_5_tensor = paddle.rand([1, 200], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
res = paddle.static.py_func(func=arg_1,x=arg_2,out=arg_3,backward_func=arg_4,skip_vars_in_backward_input=arg_5,)
