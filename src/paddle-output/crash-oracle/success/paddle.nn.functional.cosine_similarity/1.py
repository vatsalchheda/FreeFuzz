import paddle
arg_1_tensor = paddle.rand([192], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([192], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 0
arg_4 = 47.00000001
res = paddle.nn.functional.cosine_similarity(arg_1,arg_2,axis=arg_3,eps=arg_4,)
