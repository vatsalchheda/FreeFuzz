import paddle
arg_1_tensor = paddle.randint(-256,256,[3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-512,64,[3, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = "gels"
res = paddle.linalg.lstsq(arg_1,arg_2,driver=arg_3,)
