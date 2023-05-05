import paddle
arg_1_tensor = paddle.randint(0,2,[])
arg_1 = arg_1_tensor.clone()
arg_2 = "true_fn"
arg_3 = "<lambda>"
res = paddle.static.nn.cond(arg_1,arg_2,arg_3,)
