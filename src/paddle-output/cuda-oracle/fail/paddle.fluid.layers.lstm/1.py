results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 100, 256], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.rand([1, 100, 150], dtype=paddle.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.rand([1, 100, 150], dtype=paddle.float32)
arg_3 = arg_3_tensor.clone()
arg_4 = 16
arg_5 = 150
arg_6 = 1
arg_7 = 0.2
try:
  results["res_cpu"] = paddle.fluid.layers.lstm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,dropout_prob=arg_7,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.lstm(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,dropout_prob=arg_7,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
