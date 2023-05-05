import paddle
real = paddle.rand([1, 30001], paddle.float32)
imag = paddle.rand([1, 30001], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 5
res = paddle.topk(arg_1,k=arg_2,)
