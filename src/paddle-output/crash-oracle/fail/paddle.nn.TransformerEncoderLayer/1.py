import paddle
arg_1 = True
arg_2 = 2
arg_3 = 8
arg_4 = 0.1
arg_5 = "gelu"
arg_6 = 0.0
arg_7 = 0
arg_8 = True
arg_9 = None
arg_10 = None
arg_class = paddle.nn.TransformerEncoderLayer(d_model=arg_1,nhead=arg_2,dim_feedforward=arg_3,dropout=arg_4,activation=arg_5,attn_dropout=arg_6,act_dropout=arg_7,normalize_before=arg_8,weight_attr=arg_9,bias_attr=arg_10,)
arg_11_0_tensor = paddle.rand([1, 1, 8], dtype=paddle.float32)
arg_11_0 = arg_11_0_tensor.clone()
arg_11 = [arg_11_0,]
res = arg_class(*arg_11)
