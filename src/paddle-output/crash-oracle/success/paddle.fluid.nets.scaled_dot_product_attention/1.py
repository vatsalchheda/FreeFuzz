import paddle
arg_1_tensor = paddle.rand([3, 5, 9], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([3, 6, 9], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([3, 6, 10], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
res = paddle.fluid.nets.scaled_dot_product_attention(arg_1,arg_2,arg_3,)
