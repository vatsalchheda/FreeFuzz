results = dict()
import paddle
import time
arg_1 = 8
arg_2 = 2
arg_3 = 37
arg_4 = 0.1
arg_5 = "gelu"
arg_6 = 0.1
arg_7 = 0
arg_8 = True
arg_9 = None
arg_10 = None
arg_class = paddle.nn.TransformerEncoderLayer(d_model=arg_1,nhead=arg_2,dim_feedforward=arg_3,dropout=arg_4,activation=arg_5,attn_dropout=arg_6,act_dropout=arg_7,normalize_before=arg_8,weight_attr=arg_9,bias_attr=arg_10,)
arg_11_0_tensor = paddle.rand([1, 1, 8], dtype=paddle.float32)
arg_11_0 = arg_11_0_tensor.clone()
arg_11 = [arg_11_0,]
start = time.time()
results["time_low"] = arg_class(*arg_11)
results["time_low"] = time.time() - start
arg_11_0 = arg_11_0_tensor.clone().astype(paddle.float32)
arg_11 = [arg_11_0,]
start = time.time()
results["time_high"] = arg_class(*arg_11)
results["time_high"] = time.time() - start

print(results)
