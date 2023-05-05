import paddle
arg_1_tensor = paddle.rand([10, 4], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([10, 4], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([2, 21, 4], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = "decode_center_size"
res = paddle.fluid.layers.detection.box_coder(prior_box=arg_1,prior_box_var=arg_2,target_box=arg_3,code_type=arg_4,)
