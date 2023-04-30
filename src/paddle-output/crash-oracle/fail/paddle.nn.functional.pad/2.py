import paddle
arg_1_tensor = paddle.randint(-4096,64,[1, 220500, 1], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 249
arg_2_1 = 193
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = "reflect"
arg_4 = "NLC"
res = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,data_format=arg_4,)
