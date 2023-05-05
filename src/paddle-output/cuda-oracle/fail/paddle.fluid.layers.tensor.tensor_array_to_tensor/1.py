results = dict()
import paddle
arg_1_tensor = paddle.rand([-1, 128], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = -41
arg_3 = -79
try:
  results["res_cpu"] = paddle.fluid.layers.tensor.tensor_array_to_tensor(arg_1,axis=arg_2,use_stack=arg_3,)
except Exception as e:
results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = paddle.fluid.layers.tensor.tensor_array_to_tensor(arg_1,axis=arg_2,use_stack=arg_3,)
except Exception as e:
results["err_gpu"] = "ERROR:"+str(e)

print(results)
