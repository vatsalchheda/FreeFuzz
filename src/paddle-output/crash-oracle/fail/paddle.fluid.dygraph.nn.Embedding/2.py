import paddle
arg_1_0 = 19
arg_1_1 = -48
arg_1 = [arg_1_0,arg_1_1,]
arg_2 = "emb.w"
arg_3 = False
arg_class = paddle.fluid.dygraph.nn.Embedding(size=arg_1,param_attr=arg_2,is_sparse=arg_3,)
arg_4_0_tensor = paddle.randint(-32, 32768, [1], dtype=paddle.int64)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
