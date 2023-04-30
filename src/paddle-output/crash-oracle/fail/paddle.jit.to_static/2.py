import paddle
arg_1 = "forward"
res = paddle.jit.to_static(arg_1,)
