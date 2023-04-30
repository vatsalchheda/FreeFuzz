import paddle
arg_1_tensor = paddle.randint(-128,256,[257], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "L"
res = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
