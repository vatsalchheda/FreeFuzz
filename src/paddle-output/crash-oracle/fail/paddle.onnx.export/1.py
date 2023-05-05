import paddle
arg_1 = "__main__Logic"
arg_2 = 1012
arg_3_0_tensor = paddle.randint(-32,4096,[1], dtype=paddle.int64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0_tensor = paddle.randint(-4096,8192,[1], dtype=paddle.int64)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,output_spec=arg_4,)
