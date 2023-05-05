results = dict()
import paddle
arg_1_tensor = paddle.randint(-32,2048,[-1], dtype=paddle.int32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-4096,32,[1], dtype=paddle.int32)
arg_2 = arg_2_tensor.clone()
try:
  results["res_cpu"] = paddle.fluid.layers.control_flow.less_equal(arg_1,arg_2,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.control_flow.less_equal(arg_1,arg_2,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
