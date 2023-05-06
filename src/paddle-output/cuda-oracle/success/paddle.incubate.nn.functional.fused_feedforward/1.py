results = dict()
import paddle
arg_1_tensor = paddle.rand([2, 4, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([128, 512], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([512, 128], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4_tensor = paddle.rand([512], dtype=paddle.float32)
arg_4 = arg_4_tensor.clone()
arg_5_tensor = paddle.rand([128], dtype=paddle.float32)
arg_5 = arg_5_tensor.clone()
arg_6 = None
arg_7 = None
arg_8_tensor = paddle.rand([128], dtype=paddle.float32)
arg_8 = arg_8_tensor.clone()
arg_9_tensor = paddle.rand([128], dtype=paddle.float32)
arg_9 = arg_9_tensor.clone()
arg_10 = 0.0
arg_11 = 0.1
arg_12 = "relu"
arg_13 = 1e-05
arg_14 = 1e-05
arg_15 = 97
arg_16 = True
arg_17 = -1
arg_18 = None
try:
  results["res_cpu"] = paddle.incubate.nn.functional.fused_feedforward(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,dropout1_rate=arg_10,dropout2_rate=arg_11,activation=arg_12,ln1_epsilon=arg_13,ln2_epsilon=arg_14,pre_layer_norm=arg_15,training=arg_16,ring_id=arg_17,name=arg_18,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
arg_4 = arg_4_tensor.clone().cuda()
arg_5 = arg_5_tensor.clone().cuda()
arg_8 = arg_8_tensor.clone().cuda()
arg_9 = arg_9_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.incubate.nn.functional.fused_feedforward(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,arg_9,dropout1_rate=arg_10,dropout2_rate=arg_11,activation=arg_12,ln1_epsilon=arg_13,ln2_epsilon=arg_14,pre_layer_norm=arg_15,training=arg_16,ring_id=arg_17,name=arg_18,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
