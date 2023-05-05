results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 2048], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2048
arg_3 = False
try:
  results["res_cpu"] = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.dynamic_lstm(input=arg_1,size=arg_2,use_peepholes=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
