import paddle
arg_1_tensor = paddle.randint(-16384, 1, [], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = 5
arg_4 = 0
res = paddle.fluid.layers.beam_search_decode(arg_1,arg_2,beam_size=arg_3,end_id=arg_4,)
