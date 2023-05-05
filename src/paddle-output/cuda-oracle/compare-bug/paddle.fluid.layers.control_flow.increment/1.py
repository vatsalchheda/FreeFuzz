results = dict()
import paddle
arg_1_tensor = paddle.randint(-4096, 8, [1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1.0
arg_3 = True
try:
  results["res_cpu"] = paddle.fluid.layers.control_flow.increment(x=arg_1,value=arg_2,in_place=arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.control_flow.increment(x=arg_1,value=arg_2,in_place=arg_3,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)

print(results)
