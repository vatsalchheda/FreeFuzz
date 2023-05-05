import paddle
arg_1_tensor = paddle.rand([1, 8, 8], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([8, 8], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([8, 8, 1], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([8], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5_tensor = paddle.rand([8], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = None
arg_7 = None
arg_8_tensor = paddle.rand([8], dtype=paddle.float32)
arg_8 = arg_8_tensor.clone()
arg_9_tensor = paddle.rand([8], dtype=paddle.float32)
arg_9 = arg_9_tensor.clone()
arg_10 = -29.9
arg_11 = -63.0
arg_12 = "relu"
arg_13 = 1e-05
arg_14 = 15.000010000000003
arg_15 = False
arg_16 = True
arg_17 = -1
arg_18 = None
res = paddle.incubate.nn.functional.fused_feedforward(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,dropout1_rate=arg_10,dropout2_rate=arg_11,activation=arg_12,ln1_epsilon=arg_13,ln2_epsilon=arg_14,pre_layer_norm=arg_15,training=arg_16,ring_id=arg_17,name=arg_18,)
