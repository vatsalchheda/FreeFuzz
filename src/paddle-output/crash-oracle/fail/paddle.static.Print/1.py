import paddle
arg_1_tensor = paddle.randint(-512, 2048, [2, 3], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2 = "reflect"
res = paddle.static.Print(arg_1,message=arg_2,)
