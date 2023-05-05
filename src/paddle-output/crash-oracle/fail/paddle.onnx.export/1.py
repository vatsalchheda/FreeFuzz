import paddle
arg_1 = "__main__LinearNet"
arg_2 = "linear_net"
arg_3_0_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,)
