import paddle
arg_1_tensor = paddle.randint(-16384, 8, [-1, 1], dtype=paddle.int64arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([-1, 1], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16, 8, [11, 4, 4], dtype=paddle.int64arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([-1, 4], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5 = True
arg_6 = -16
res = paddle.fluid.layers.beam_search(pre_ids=arg_1,pre_scores=arg_2,ids=arg_3,scores=arg_4,beam_size=arg_5,end_id=arg_6,)
