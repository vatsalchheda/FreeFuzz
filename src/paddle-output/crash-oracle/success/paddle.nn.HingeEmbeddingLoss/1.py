import paddle
arg_1 = 1.0
arg_2 = "none"
arg_class = paddle.nn.HingeEmbeddingLoss(margin=arg_1,reduction=arg_2,)
arg_3_0_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,]
res = arg_class(*arg_3)
