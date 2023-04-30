import paddle
arg_1_tensor = paddle.randint(-8,2048,[5, 5], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "reflect"
arg_3 = True
arg_4 = None
res = paddle.nansum(arg_1,axis=arg_2,keepdim=arg_3,name=arg_4,)
