import paddle
arg_1 = "__main__Mnist"
arg_2 = "inference_model"
arg_3_0_tensor = paddle.rand([-1, 784], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = paddle.jit.save(arg_1,arg_2,input_spec=arg_3,)
