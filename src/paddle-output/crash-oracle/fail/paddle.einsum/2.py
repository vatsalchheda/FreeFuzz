import paddle
arg_1 = "...jk->...kj"
arg_2_tensor = paddle.randint(-2048,256,[2, 3, 2], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
res = paddle.einsum(arg_1,arg_2,)
