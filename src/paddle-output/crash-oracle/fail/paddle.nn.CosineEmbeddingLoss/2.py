import paddle
arg_1 = 60.5
arg_2 = "mean"
arg_class = paddle.nn.CosineEmbeddingLoss(margin=arg_1,reduction=arg_2,)
arg_3_0_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.rand([2, 3], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.randint(-4096, 256, [2], dtype=paddle.int64)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
res = arg_class(*arg_3)
