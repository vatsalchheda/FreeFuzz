results = dict()
import paddle
import time
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
float_tensor = paddle.rand([1, 1, 8], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_11_0_tensor = f16_tensor
arg_11_0 = arg_11_0_tensor.clone()
arg_11 = [arg_11_0,]
start = time.time()
results["time_low"] = arg_class(*arg_11)
results["time_low"] = time.time() - start
arg_11_0 = arg_11_0_tensor.clone().type(paddle.float32)
arg_11 = [arg_11_0,]
start = time.time()
results["time_high"] = arg_class(*arg_11)
results["time_high"] = time.time() - start

print(results)
