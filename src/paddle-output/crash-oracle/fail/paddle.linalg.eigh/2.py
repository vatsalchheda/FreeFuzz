import paddle
arg_1_tensor = paddle.randint(-1024,16,[2, 2], dtype=paddle.complex64)
arg_1 = arg_1_tensor.clone()
arg_2 = "L"
res = paddle.linalg.eigh(arg_1,UPLO=arg_2,)
