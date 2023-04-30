import paddle
arg_1 = 0.5
arg_2 = "mean"
arg_class = paddle.nn.CosineEmbeddingLoss(margin=arg_1,reduction=arg_2,)
arg_3_0_tensor = paddle.randint(-2,8,[2, 3], dtype=paddle.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3_1_tensor = paddle.randint(-8,1,[2, 3], dtype=paddle.float32)
arg_3_1 = arg_3_1_tensor.clone()
arg_3_2_tensor = paddle.randint(-4,64,[2], dtype=paddle.int64)
arg_3_2 = arg_3_2_tensor.clone()
arg_3 = [arg_3_0,arg_3_1,arg_3_2,]
res = arg_class(*arg_3)
