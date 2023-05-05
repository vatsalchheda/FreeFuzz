import paddle
arg_1_tensor = paddle.rand([1, 1, 182, 182], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_0 = False
arg_2_1 = "max"
arg_2_2 = True
arg_2_3 = "reflect"
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
arg_3 = "replicate"
arg_4 = 0.0
arg_5 = "NCHW"
arg_6 = None
res = paddle.nn.functional.pad(arg_1,pad=arg_2,mode=arg_3,value=arg_4,data_format=arg_5,name=arg_6,)
