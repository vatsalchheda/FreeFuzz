import paddle
arg_1_tensor = paddle.randint(-32,1024,[2, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-1024,1,[2, 3], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-2048,32768,[3, 3], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 0.5
arg_5 = "mean"
arg_6 = None
res = paddle.nn.functional.cosine_embedding_loss(arg_1,arg_2,arg_3,margin=arg_4,reduction=arg_5,name=arg_6,)
