import paddle
arg_1_tensor = paddle.rand([2, 3, 4], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 3, 4], dtype=paddle.float64)
arg_2 = arg_2_tensor.clone()
arg_3_0 = "reflect"
arg_3_1 = -1e+20
arg_3 = [arg_3_0,arg_3_1,]
res = paddle.tensordot(arg_1,arg_2,axes=arg_3,)
