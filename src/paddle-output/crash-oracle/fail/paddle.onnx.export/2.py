import paddle
arg_1 = "__main__Logic"
arg_2 = 10.0
arg_3_0_tensor = paddle.randint(-256, 256, [1], dtype=paddle.int64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
arg_4_0_tensor = paddle.randint(-8192, 1, [1], dtype=paddle.int64)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = paddle.onnx.export(arg_1,arg_2,input_spec=arg_3,output_spec=arg_4,)
