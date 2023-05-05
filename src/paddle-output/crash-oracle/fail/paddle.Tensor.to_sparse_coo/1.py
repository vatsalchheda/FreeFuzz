import paddle
real = paddle.rand([], paddle.float32)
imag = paddle.rand([], paddle.float32)
arg_1_tensor = paddle.complex(real, imag)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
res = paddle.Tensor.to_sparse_coo(arg_1,arg_2,)
